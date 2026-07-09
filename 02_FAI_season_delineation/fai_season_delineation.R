# ================================================================
# Supplementary analysis: FAI-based fire-season delineation
# Purpose:
#   1) Construct a Fire Activity Index (FAI) from Fire Count and Burned Area.
#   2) Identify three major seasonal fire-activity peaks objectively.
#   3) Export a TIFF supplementary figure and threshold-sensitivity tables.
#
# The script reads the fire-event and burned-pixel event tables directly.
# ================================================================

rm(list = ls())

library(dplyr)
library(ggplot2)
library(mgcv)

# ------------------------------------------------
# 1. Configuration
# ------------------------------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(
      sub("^--file=", "", file_arg[1]),
      winslash = "/",
      mustWork = TRUE
    )))
  }
  
  frame_files <- vapply(
    sys.frames(),
    function(frame) {
      if (!is.null(frame$ofile)) as.character(frame$ofile) else NA_character_
    },
    character(1)
  )
  frame_files <- frame_files[!is.na(frame_files) & nzchar(frame_files)]
  if (length(frame_files) > 0) {
    return(dirname(normalizePath(
      tail(frame_files, 1),
      winslash = "/",
      mustWork = TRUE
    )))
  }
  
  if (
    requireNamespace("rstudioapi", quietly = TRUE) &&
    rstudioapi::isAvailable()
  ) {
    editor_path <- rstudioapi::getSourceEditorContext()$path
    if (nzchar(editor_path)) {
      return(dirname(normalizePath(
        editor_path,
        winslash = "/",
        mustWork = TRUE
      )))
    }
  }
  
  stop(
    "Cannot determine the script location. ",
    "Run the whole file with RStudio Source or Rscript."
  )
}

script_dir <- get_script_dir()
out_dir <- file.path(script_dir, "output")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

code_root <- normalizePath(
  file.path(script_dir, ".."),
  winslash = "/",
  mustWork = TRUE
)
input_dir <- file.path(code_root, "00_Input_data")
fire_event_file <- file.path(
  input_dir,
  "Fire_Event_Table_2001_2025.csv"
)
ba_event_file <- file.path(
  input_dir,
  "Burned_Pixel_Event_Table_2001_2025.csv"
)

if (!file.exists(fire_event_file)) {
  stop("Canonical fire event table not found: ", fire_event_file)
}
if (!file.exists(ba_event_file)) {
  stop("Canonical burned-pixel event table not found: ", ba_event_file)
}

# Use the installed Times New Roman font directly.
# This is compatible with the Cairo TIFF device used below.
font_family <- "Times New Roman"
if (.Platform$OS.type == "windows") {
  try(
    do.call(
      grDevices::windowsFonts,
      setNames(
        list(grDevices::windowsFont("Times New Roman")),
        font_family
      )
    ),
    silent = TRUE
  )
}

# Final fire-prone seasons used in the manuscript.
# These are retained to keep consistency with the main seasonal analyses.
season_df <- data.frame(
  Season = c("Spring", "Summer", "Autumn"),
  start  = c(44, 161, 269),
  end    = c(160, 218, 331),
  fill   = c("#B3DE69", "#FB8072", "#FFED6F")
)

# Thresholds used for sensitivity analysis.
# The 3% threshold is intentionally excluded because the supplementary analysis
# focuses on core fire-activity windows rather than very low-intensity tails.
threshold_levels <- c(0.05, 0.08, 0.10)

# Parameters for smoothing and peak detection.
# k = 30 is used in the cyclic GAM to preserve broad seasonal structure.
gam_k <- 30
min_peak_distance <- 60
peak_height_ratio <- 0.05

# ------------------------------------------------
# 2. Helper functions
# ------------------------------------------------

# Convert DOY to English calendar date using a non-leap-year calendar.
# This matches the manuscript date ranges: DOY 44 = Feb 13.
# Use a locale-independent calendar conversion for reproducible date labels.
doy_to_date <- function(doy) {
  doy <- as.integer(round(doy))
  month_names <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
  month_days <- c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
  
  if (is.na(doy)) return(NA_character_)
  if (doy < 1) return(NA_character_)
  if (doy > 365) return("Dec 31")
  
  cum_days <- cumsum(month_days)
  month_id <- which(doy <= cum_days)[1]
  day_in_month <- doy - c(0, cum_days)[month_id]
  
  paste0(month_names[month_id], " ", day_in_month)
}

format_doy_range <- function(start, end) {
  paste0("DOY ", start, "-", end, " (", doy_to_date(start), "-", doy_to_date(end), ")")
}

# Extract continuous intervals from a logical vector.
get_intervals <- function(active) {
  active <- as.logical(as.vector(unlist(active)))
  active[is.na(active)] <- FALSE
  
  r <- rle(active)
  ends <- cumsum(r$lengths)
  starts <- ends - r$lengths + 1
  
  if (!any(r$values)) {
    return(data.frame(start = integer(0), end = integer(0)))
  }
  
  data.frame(
    start = starts[r$values],
    end = ends[r$values]
  )
}

# Keep peaks with a minimum circular distance.
keep_peaks <- function(peaks, y, min_dist = 60, n = 366) {
  if (length(peaks) == 0) return(integer(0))
  
  peaks_ordered <- peaks[order(y[peaks], decreasing = TRUE)]
  kept <- c()
  
  for (p in peaks_ordered) {
    if (length(kept) == 0) {
      kept <- c(kept, p)
    } else {
      circular_dist <- pmin(abs(p - kept), n - abs(p - kept))
      if (all(circular_dist >= min_dist)) {
        kept <- c(kept, p)
      }
    }
  }
  
  sort(kept)
}

# Assign each threshold-derived interval to one of the final seasons
# according to its midpoint.
assign_season <- function(start, end, season_df) {
  mid <- (start + end) / 2
  
  idx <- which(mid >= season_df$start & mid <= season_df$end)
  if (length(idx) == 0) return("Transition")
  season_df$Season[idx[1]]
}

# Combine intervals into a compact season-wise table.
collapse_intervals_by_season <- function(interval_df, season_df) {
  out <- data.frame(
    Spring = "-",
    Summer = "-",
    Autumn = "-"
  )
  
  if (nrow(interval_df) == 0) return(out)
  
  interval_df$Season <- mapply(assign_season, interval_df$start, interval_df$end,
                               MoreArgs = list(season_df = season_df))
  interval_df$Label <- mapply(format_doy_range, interval_df$start, interval_df$end)
  
  for (ss in c("Spring", "Summer", "Autumn")) {
    labels <- interval_df$Label[interval_df$Season == ss]
    if (length(labels) > 0) {
      out[[ss]] <- paste(labels, collapse = "; ")
    }
  }
  
  out
}

# ------------------------------------------------
# 3. Read canonical events and construct daily Fire Activity Index
# ------------------------------------------------

fire_events <- read.csv(
  fire_event_file,
  fileEncoding = "UTF-8-BOM",
  check.names = FALSE,
  stringsAsFactors = FALSE
)
ba_events <- read.csv(
  ba_event_file,
  fileEncoding = "UTF-8-BOM",
  check.names = FALSE,
  stringsAsFactors = FALSE
)

required_fire <- c("Fire_ID", "DOY")
required_ba <- c("Burned_Pixel_ID", "DOY", "Burned_Area_km2")
missing_fire <- setdiff(required_fire, names(fire_events))
missing_ba <- setdiff(required_ba, names(ba_events))
if (length(missing_fire) > 0) {
  stop(
    "Canonical fire event table is missing columns: ",
    paste(missing_fire, collapse = ", ")
  )
}
if (length(missing_ba) > 0) {
  stop(
    "Canonical BA event table is missing columns: ",
    paste(missing_ba, collapse = ", ")
  )
}

fc_daily <- fire_events %>%
  group_by(DOY) %>%
  summarise(Fire_Count = n(), .groups = "drop") %>%
  rename(Day_number = DOY)

ba_daily <- ba_events %>%
  group_by(DOY) %>%
  summarise(
    Burned_Area = sum(Burned_Area_km2, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  rename(Day_number = DOY)

# Ensure DOY 1-366 is complete. Missing days are filled with zero.
all_days <- data.frame(Day_number = 1:366)
daily <- all_days %>%
  left_join(fc_daily, by = "Day_number") %>%
  left_join(ba_daily, by = "Day_number") %>%
  mutate(
    Fire_Count = ifelse(is.na(Fire_Count), 0, Fire_Count),
    Burned_Area = ifelse(is.na(Burned_Area), 0, Burned_Area)
  )

# Normalize Fire Count and Burned Area to 0-1 and construct FAI.
daily <- daily %>%
  mutate(
    FC_norm = Fire_Count / max(Fire_Count, na.rm = TRUE),
    BA_norm = Burned_Area / max(Burned_Area, na.rm = TRUE),
    FAI = (FC_norm + BA_norm) / 2
  )

# ------------------------------------------------
# 4. Smooth FAI using cyclic GAM
# ------------------------------------------------

m_fai <- gam(
  FAI ~ s(Day_number, bs = "cc", k = gam_k),
  data = daily,
  method = "REML",
  knots = list(Day_number = c(1, 366))
)

daily$FAI_smooth <- predict(m_fai, newdata = daily)
daily$FAI_smooth <- pmax(as.numeric(as.vector(unlist(daily$FAI_smooth))), 0)

# ------------------------------------------------
# 5. Identify major FAI peaks
# ------------------------------------------------

y <- daily$FAI_smooth
n <- length(y)

is_peak <- y > c(y[n], y[-n]) & y > c(y[-1], y[1])
peak_candidates <- which(is_peak)

height_threshold <- peak_height_ratio * max(y, na.rm = TRUE)
peak_candidates <- peak_candidates[y[peak_candidates] >= height_threshold]

peaks <- keep_peaks(
  peaks = peak_candidates,
  y = y,
  min_dist = min_peak_distance,
  n = 366
)

# Keep the three strongest peaks, then sort chronologically.
if (length(peaks) > 3) {
  peaks <- peaks[order(y[peaks], decreasing = TRUE)][1:3]
  peaks <- sort(peaks)
}

peak_df <- daily[peaks, ]
peak_df$Peak <- c("Spring peak", "Summer peak", "Autumn peak")[seq_along(peaks)]
peak_df$Date <- vapply(peak_df$Day_number, doy_to_date, character(1))

# Export peak results.
write.csv(
  peak_df[, c("Peak", "Day_number", "Date", "FAI_smooth")],
  file.path(out_dir, "Supplementary_FAI_peak_results.csv"),
  row.names = FALSE
)

# ------------------------------------------------
# 6. Threshold-based sensitivity analysis
# ------------------------------------------------

sensitivity_long <- data.frame()
sensitivity_wide <- data.frame(
  Method = "Final season used in this study",
  Spring = format_doy_range(44, 160),
  Summer = format_doy_range(161, 218),
  Autumn = format_doy_range(269, 331),
  Notes = "Final seasonal windows used for all seasonal analyses; these windows include the core FAI peaks and adjacent low-intensity transitional periods."
)

for (thr in threshold_levels) {
  threshold_value <- thr * max(daily$FAI_smooth, na.rm = TRUE)
  active <- daily$FAI_smooth >= threshold_value
  intervals <- get_intervals(active)
  
  if (nrow(intervals) > 0) {
    intervals$threshold <- thr
    intervals$threshold_label <- paste0(thr * 100, "% of maximum smoothed FAI")
    intervals$Season <- mapply(assign_season, intervals$start, intervals$end,
                               MoreArgs = list(season_df = season_df))
    intervals$Date_range <- mapply(format_doy_range, intervals$start, intervals$end)
    sensitivity_long <- rbind(sensitivity_long, intervals)
  }
  
  collapsed <- collapse_intervals_by_season(intervals, season_df)
  
  sensitivity_wide <- rbind(
    sensitivity_wide,
    data.frame(
      Method = paste0(thr * 100, "% FAI threshold"),
      Spring = collapsed$Spring,
      Summer = collapsed$Summer,
      Autumn = collapsed$Autumn,
      Notes = "Threshold-derived core fire-activity windows based on the smoothed FAI curve. Higher thresholds identify narrower core periods."
    )
  )
}

# Add peak row to the final supplementary table.
peak_row <- data.frame(
  Method = "Major FAI peak",
  Spring = ifelse(length(peaks) >= 1,
                  paste0("DOY ", peaks[1], " (", doy_to_date(peaks[1]), ")"), "-"),
  Summer = ifelse(length(peaks) >= 2,
                  paste0("DOY ", peaks[2], " (", doy_to_date(peaks[2]), ")"), "-"),
  Autumn = ifelse(length(peaks) >= 3,
                  paste0("DOY ", peaks[3], " (", doy_to_date(peaks[3]), ")"), "-"),
  Notes = "Major local maxima identified from the smoothed FAI curve using the minimum peak-distance criterion."
)

sensitivity_wide <- rbind(sensitivity_wide, peak_row)

write.csv(
  sensitivity_long,
  file.path(out_dir, "Supplementary_Table_S2_FAI_sensitivity_long.csv"),
  row.names = FALSE
)

write.csv(
  sensitivity_wide,
  file.path(out_dir, "Supplementary_Table_S2_FAI_sensitivity.csv"),
  row.names = FALSE
)

# ------------------------------------------------
# 7. Supplementary figure: FAI curve, peaks, and retained seasons
# ------------------------------------------------

# Theme consistent with the manuscript temporal plot.
# linewidth is used instead of size for lines to avoid ggplot2 >= 3.4 warnings.
base_theme <- theme(
  text = element_text(family = font_family),
  panel.grid.major = element_line(linewidth = 0.3),
  panel.grid.minor = element_line(colour = NA),
  legend.title = element_text(family = font_family, face = "bold", size = 18),
  legend.text = element_text(family = font_family, size = 16),
  axis.title = element_text(family = font_family, size = 18),
  axis.text = element_text(family = font_family, size = 14),
  plot.title = element_text(family = font_family, size = 20),
  plot.subtitle = element_text(family = font_family, size = 14),
  strip.text = element_text(family = font_family, size = 16)
)

threshold_df <- data.frame(
  threshold = threshold_levels,
  yintercept = threshold_levels * max(daily$FAI_smooth, na.rm = TRUE),
  label = paste0(threshold_levels * 100, "% threshold")
)

p_fai <- ggplot(daily, aes(x = Day_number)) +
  annotate("rect", xmin = 43.5, xmax = 160.5, ymin = -Inf, ymax = Inf,
           fill = "#B3DE69", alpha = 0.3) +
  annotate("rect", xmin = 160.5, xmax = 218.5, ymin = -Inf, ymax = Inf,
           fill = "#FB8072", alpha = 0.3) +
  annotate("rect", xmin = 268.5, xmax = 331.5, ymin = -Inf, ymax = Inf,
           fill = "#FFED6F", alpha = 0.3) +
  geom_line(aes(y = FAI, color = "Daily FAI"), linewidth = 0.45, alpha = 0.55) +
  geom_line(aes(y = FAI_smooth, color = "Smoothed FAI"), linewidth = 1.0) +
  geom_hline(data = threshold_df,
             aes(yintercept = yintercept, linetype = label),
             color = "#666666", linewidth = 0.45) +
  geom_point(data = peak_df, aes(y = FAI_smooth), size = 2.5, color = "black") +
  geom_text(data = peak_df,
            aes(y = FAI_smooth, label = paste0("DOY ", Day_number)),
            vjust = -1.0, hjust = 0.5,
            family = font_family, size = 4.5, color = "#333333") +
  annotate("text", x = 44.5, y = max(daily$FAI_smooth) * 0.95,
           label = "Spring\nFeb 13-Jun 9", vjust = 1, hjust = 0,
           color = "#666666", size = 5, family = font_family) +
  annotate("text", x = 161.5, y = max(daily$FAI_smooth) * 0.95,
           label = "Summer\nJun 10-Aug 6", vjust = 1, hjust = 0,
           color = "#666666", size = 5, family = font_family) +
  annotate("text", x = 269.5, y = max(daily$FAI_smooth) * 0.95,
           label = "Autumn\nSep 26-Nov 27", vjust = 1, hjust = 0,
           color = "#666666", size = 5, family = font_family) +
  scale_x_continuous(breaks = seq(0, 366, 30), expand = c(0.002, 0)) +
  scale_y_continuous(expand = c(0.002, 0)) +
  scale_color_manual(
    name = "FAI curve",
    values = c("Daily FAI" = "grey70", "Smoothed FAI" = "black")
  ) +
  scale_linetype_manual(
    name = "Sensitivity threshold",
    values = c("5% threshold" = "dashed",
               "8% threshold" = "dotdash",
               "10% threshold" = "dotted")
  ) +
  labs(
    x = "Day of year",
    y = "Fire Activity Index (FAI)",
    title = "FAI-based fire-season delineation"
  ) +
  base_theme

# Export the supplementary figure as TIFF only. On Windows, use the
# native graphics device so Times New Roman is resolved through the Windows
# font database rather than through Cairo's PDF CID font lookup.
tiff_file <- file.path(
  out_dir,
  "Supplementary_Figure_S1_FAI_season_delineation.tif"
)
tiff_args <- list(
  filename = tiff_file,
  width = 9,
  height = 5,
  units = "in",
  res = 600,
  compression = "lzw"
)
if (.Platform$OS.type == "windows") {
  tiff_args$type <- "windows"
} else if (capabilities("cairo")) {
  tiff_args$type <- "cairo"
}
do.call(grDevices::tiff, tiff_args)
print(p_fai)
dev.off()

# ------------------------------------------------
# 8. Console summary
# ------------------------------------------------

cat("\n============================================================\n")
cat("FAI-based fire-season delineation completed.\n")
cat("Output folder:", out_dir, "\n\n")

cat("Major FAI peaks:\n")
print(peak_df[, c("Peak", "Day_number", "Date", "FAI_smooth")])

cat("\nSensitivity table:\n")
print(sensitivity_wide)

cat("\nFiles exported:\n")
cat("\n - Supplementary_Figure_S1_FAI_season_delineation.tif")
cat("\n - Supplementary_Table_S2_FAI_sensitivity.csv")
cat("\n - Supplementary_Table_S2_FAI_sensitivity_long.csv")
cat("\n - Supplementary_FAI_peak_results.csv\n")
cat("============================================================\n")
