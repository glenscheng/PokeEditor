import os
import subprocess
import argparse
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict
import ffmpeg


import subprocess
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos as _orig_parse

def ffmpeg_parse_infos(*args, **kwargs):
    # MoviePy may pass check_duration, print_infos, etc. — we only care about filename
    filename = args[0] if args else kwargs.get("filename")
    # remove the extra keys so _orig_parse never sees them
    kwargs.pop("check_duration", None)
    print_infos = kwargs.pop("print_infos", False)

    # run ffmpeg -i to capture stderr
    result = subprocess.run(
        ["ffmpeg", "-i", filename],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True
    )
    # strip the final “must be specified” line
    cleaned = "\n".join(
        line for line in result.stderr.splitlines()
        if not line.startswith("At least one output file must be specified")
    )

    # call the original parser, passing the cleaned stderr log as infos
    return _orig_parse(filename, infos=cleaned, print_infos=print_infos)

# install our wrapper into MoviePy
import moviepy.video.io.ffmpeg_reader as reader_mod
reader_mod.ffmpeg_parse_infos = ffmpeg_parse_infos


from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip



# CSV_PATH = '../data/prices.csv'

# Global definitions
REVEAL_SOUND_PATH = '../sounds/chaching.mp3'
MAX_REVEALS = 10 # energy card doesn't count for this
FRAME_SKIP = 3 # for video processing


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pokémon Pack Video Editor")
    parser.add_argument('--video',  required=True, help="Path to the input video")
    parser.add_argument('--set',    required=True, help="Name of the Pokémon card set")
    parser.add_argument('--cost',   required=True, type=float, help="Cost of the pack")
    parser.add_argument('--day',    required=True, type=int,   help="Day number")
    parser.add_argument(
        '--prices',
        required=True,
        help="Comma-separated list of 11 card prices: first for pre-first reveal, then one per reveal"
    )
    return parser.parse_args()

# Helper functions for detection

def compute_motion_diff(prev_gray, gray):
    diff = cv2.absdiff(prev_gray, gray)
    blur = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)
    non_zero = np.count_nonzero(thresh)
    return thresh, non_zero

def get_skin_ratio(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h*0.5):h, int(w*0.25):int(w*0.75)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([0, 30, 150], dtype=np.uint8),
                       np.array([20, 150, 255], dtype=np.uint8))
    return np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])

def get_gray_ratio(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h*0.3):int(h*0.6), 0:w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, 180, 255)
    return np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])

def get_blue_ratio(frame):
    h, w = frame.shape[:2]
    roi = frame[int(h*0.5):h, int(w*0.25):int(w*0.75)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,
                       np.array([95, 10, 200], dtype=np.uint8),
                       np.array([135, 100, 255], dtype=np.uint8))
    return np.count_nonzero(mask) / (roi.shape[0] * roi.shape[1])

def is_card_reveal(thresh, last_ts, timestamp, gap, motion_threshold, debug=False):
    h, w = thresh.shape
    mask = np.zeros_like(thresh)
    cv2.rectangle(mask, (w//4, h//2), (w*3//4, h), 255, -1)
    motion = np.count_nonzero(cv2.bitwise_and(thresh, mask))
    if debug:
        print(f"motion_in_region = {motion}")
    if motion > motion_threshold and (last_ts is None or timestamp - last_ts > gap):
        return True
    return False

# Detection pipeline

def detect_card_reveals(video_path, debug=False):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    timestamps = []
    frame_count = 0

    # Limits
    MOTION_STABLE_THRESHOLD = 1020000
    MOTION_STABLE_REQUIRED_FRAMES = 27 / FRAME_SKIP
    MOTION_DETECTION_THRESHOLD = 744000
    BUFFER_TIME_BETWEEN_REVEALS = 0.8
    SKIN_THRESHOLD = 0.72
    GRAY_THRESHOLD = 0.5
    BLUE_THRESHOLD = 0.2
    MAX_BLUE_CHECKS = 15
    MAX_GRAY_CHECKS = 80

    # Counters
    motion_stable_frames = 0
    blue_checks = 0
    gray_checks = 0
    reveal_mode = False

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) # WA
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        if frame_count % FRAME_SKIP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # skip initial packaging
            if timestamp < 15:
                prev_gray = gray
                frame_count += 1
                continue

            if prev_gray is not None:
                thresh, non_zero = compute_motion_diff(prev_gray, gray)
                if debug:
                    print(f"Frame {frame_count} at {timestamp:.2f}s: non_zero = {non_zero}")

                if debug:
                    debug_dir = "debug_frames"
                    os.makedirs(debug_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(debug_dir, f"frame_{frame_count}.jpg"), frame)
                    cv2.imshow("Debug Frame", frame)
                    cv2.waitKey(1)

                if not reveal_mode:
                    if non_zero < MOTION_STABLE_THRESHOLD:
                        skin_ratio = get_skin_ratio(frame)
                        if debug:
                            print(f"Skin ratio in ROI = {skin_ratio:.2f}")

                        gray_ratio = get_gray_ratio(frame)
                        if debug:
                            print(f"Gray ratio in ROI = {gray_ratio:.2f}")

                        if blue_checks < MAX_BLUE_CHECKS:
                            blue_ratio = get_blue_ratio(frame)
                            blue_checks += 1
                            if debug:
                                print(f"Blue ratio check #{blue_checks}: {blue_ratio:.2f}")
                        else:
                            blue_ratio = 0.0

                        # stability decision
                        if skin_ratio > SKIN_THRESHOLD:
                            if debug:
                                print("[INFO] Too much hand detected; not counting as stable frame.")
                        elif gray_ratio > GRAY_THRESHOLD and gray_checks < MAX_GRAY_CHECKS:
                            gray_checks += 1
                            if debug:
                                print(f"[INFO] Too much gray detected #{gray_checks}; not counting as stable frame.")
                        elif blue_ratio > BLUE_THRESHOLD and blue_checks <= MAX_BLUE_CHECKS:
                            if debug:
                                print(f"[INFO] Too much blue detected #{blue_checks}; not counting as stable frame.")
                        else:
                            motion_stable_frames += 1
                            if debug:
                                print(f"[INFO] Motion stable at {timestamp:.2f}s. Count = {motion_stable_frames}")
                            if motion_stable_frames >= MOTION_STABLE_REQUIRED_FRAMES:
                                reveal_mode = True
                                if debug:
                                    print(f"[INFO] Stack ready at {timestamp:.2f}s. Entering reveal mode.")
                    else:
                        motion_stable_frames = 0
                else:
                    last_ts = timestamps[-1] if timestamps else None
                    if is_card_reveal(thresh, last_ts, timestamp, BUFFER_TIME_BETWEEN_REVEALS, MOTION_DETECTION_THRESHOLD, debug):
                        timestamps.append(timestamp)
                        if debug:
                            print(f"[DETECTED] Card reveal at {timestamp:.2f}s")
                        if len(timestamps) >= MAX_REVEALS:
                            break

            prev_gray = gray

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    return timestamps

def extract_frame(video_path, timestamp):
    """Extract a frame from the video at the given timestamp (in seconds)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    else:
        return None

def convert_mov_to_mp4(input_file, output_file):
    # Build the ffmpeg command. Note that the output file is the last argument.
    command = [
        'ffmpeg', '-y', '-i', input_file,
        '-map_metadata', '-1',   # strip all metadata
        '-map', '0:v:0', '-map', '0:a:0',   # only keep the first video & audio streams
        '-codec', 'copy',        # fast remux
        output_file
    ]
    
    # Remove output file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    try:
        print("Running ffmpeg command:", ' '.join(command))
        subprocess.run(command, check=True, capture_output=True)
        print(f"Conversion successful: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
         print(f"Error during conversion: {e}")
         print(f"FFmpeg output: {e.stderr.decode()}")

# Overlay functions

def ffprobe_color_metadata(path: str) -> Dict[str, str]:
    """Return color-related metadata for stream 0 as a dict."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=pix_fmt,color_range,color_primaries,color_trc,colorspace",
        "-of", "default=noprint_wrappers=1:nokey=0", path
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)

def color_flags(meta: Dict[str, str], codec: str) -> list[str]:
    """Build ffmpeg parameters that keep the original color + range."""
    # Defaults if any tag is missing
    prim  = meta.get("color_primaries", "bt709")
    trc   = meta.get("color_trc", "bt709")
    csp   = meta.get("colorspace", "bt709")
    rng   = meta.get("color_range", "tv")      # 'tv' = limited, 'pc' = full
    flags = [
        "-colorspace", csp,
        "-color_primaries", prim,
        "-color_trc", trc,
        "-color_range", rng,
    ]

    # x264/x265 need a codec-specific flag for full-range
    if rng == "pc":
        if codec == "libx264":
            flags += ["-x264-params", "fullrange=1"]
        else:  # libx265
            flags += ["-x265-params", "range=full"]

    return flags

def drawtext_filter(text, fontsize, color, xpos, ypos, enable):
    # Wrap the literal part once so we don’t repeat ourselves
    return (
        "drawtext="
        "fontfile=../font/eras-itc-bold.ttf:" # this is the font I used in Adobe Premiere Pro
        f"text='{text}':"
        f"fontsize={fontsize}:"
        f"fontcolor={color}:"
        f"x={xpos}:y={ypos}:"
        f"enable='{enable}'"
    )
    
def create_text_clips(prices, all_timestamps, text_timing):
    """
    Overlay `text` on an iPhone video without altering any picture data.
    We pull the source's color metadata with ffprobe, then ask ffmpeg
    to draw the text while copying *every* color-related tag verbatim.
    """
    # # ------------------------------------------------------------------
    # # 1.  Read the source-file color tags so we can copy them verbatim
    # # ------------------------------------------------------------------
    # meta       = ffprobe_color_metadata(input_path)
    # pix_fmt_in = meta.get("pix_fmt", "yuv420p")
    # trc        = meta.get("color_trc", "bt709")
    # prim       = meta.get("color_primaries", "bt709")
    # csp        = meta.get("colorspace", "bt709")
    # rng        = meta.get("color_range", "tv")          # tv = limited, pc = full

    # # 10-bit HDR (HLG/PQ) clips need x265 + a 10-bit pixel-format
    # is_10bit   = "10" in pix_fmt_in or trc.startswith(("smpte2084", "arib-std-b67"))
    # codec      = "libx265" if is_10bit else "libx264"
    # pix_fmt    = "yuv420p10le" if is_10bit else "yuv420p"

    # ------------------------------------------------------------------
    # 2.  Build the drawtext filter string
    # ------------------------------------------------------------------
    # Position helpers --------------------------------------------------
    # xpos = {
    #     "center": "(w-text_w)/2",          # horizontal centre
    #     "left"  : "50",
    #     "right" : "(w-text_w)-50"
    # }[position if position in ("center", "left", "right") else "center"]

    # ypos = {
    #     "center": "(h-text_h)/2",
    #     "top"   : "50",
    #     "bottom": "(h-text_h)-50"
    # }[position if position in ("center", "top", "bottom") else "center"]

    # t_end   = end_time if end_time is not None else "N"            # N = clip duration
    # enable  = f"between(t\\,{start_time}\\,{t_end})"

    fontsize = 175
    
    xpos_center = "(w-text_w)/2"
    ypos_centerish = "(h-text_h)/2-500"
    individual_prices = []
    for i, price in enumerate(prices):
        individual_prices.append(
            drawtext_filter(
                text=f"+${price:.2f}",
                fontsize=fontsize,
                color="lightgreen",
                xpos=xpos_center,
                ypos=ypos_centerish,
                enable=f"between(t,{all_timestamps[i*2]+text_timing[i*2]},{all_timestamps[i*2+1]+text_timing[i*2+1]})"
            )
        )
    
    xpos_left = "200"
    ypos_middle_top = "(h-text_h)/2-1200"
    accumulating_prices = []
    running_total = 0
    for i, price in enumerate(prices):
        running_total += price
        accumulating_prices.append(
            drawtext_filter(
                text=f"${running_total:.2f}",
                fontsize=fontsize,
                color="lightgreen",
                xpos=xpos_left,
                ypos=ypos_middle_top,
                enable=f"between(t,{all_timestamps[i*2]+text_timing[i*2]},{all_timestamps[i*2+1]+text_timing[i*2+1]})"
            )
        )
    
    draw = []
    draw = ",".join(individual_prices + accumulating_prices)
    
    return draw

        # f"drawtext="
        # f"fontfile=/System/Library/Fonts/Supplemental/Arial.ttf:" # find the same font that I used
        # f"text='{text}':"
        # f"fontsize={font_size}:"
        # f"fontcolor={color}:"
        # f"x={xpos}:y={ypos}:"
        # f"enable='{enable}'"
    

    # ------------------------------------------------------------------
    # 3.  Launch ffmpeg --------------------------------------------------
    # ------------------------------------------------------------------
    # cmd = [
    #     "ffmpeg", "-y",
    #     "-threads", "0", # 0 = auto-detect max
    #     "-i", input_path,
    #     "-vf", draw,                        # overlay the text
    #     "-c:v", codec,
    #     "-pix_fmt", pix_fmt,
    #     "-preset", "ultrafast",
    #     "-crf", "35",
    #     "-c:a", "copy",                     # copy audio bit-perfect
    #     # keep the original color tags verbatim
    #     "-colorspace",   csp,
    #     "-color_primaries", prim,
    #     "-color_trc",    trc,
    #     "-color_range",  rng,
    #     output_path
    # ]

    # print(" ".join(cmd))                   # handy for debugging
    # print()
    # subprocess.run(cmd, check=True)

# def create_audio_clips(timestamps):
#     sfx = AudioFileClip(REVEAL_SOUND_PATH).volumex(0.5)
#     return [sfx.set_start(ts+0.3) for ts in timestamps]

def create_audio_clips_command(input_video, prices, all_timestamps, text_timing):
    audio_timing = -0.5
    
    cmd = [
        "ffmpeg", "-y",
        "-threads", "0",
        "-i", input_video,
    ]
    # Add an SFX input for each timestamp
    for _ in prices:
        cmd += ["-i", REVEAL_SOUND_PATH]

    # Filter complex parts
    filter_lines = []
    # Each SFX input gets a delayed audio stream
    for i, _ in enumerate(prices):
        # SFX inputs start at 1 (0 is the main video)
        delay_ms = int((all_timestamps[i*2] + text_timing[i*2] + audio_timing) * 1000)
        filter_lines.append(f"[{i+1}]adelay={delay_ms}|{delay_ms}[sfx{i+1}]")
    # Now build amix
    sfx_labels = ''.join(f"[sfx{i+1}]" for i in range(len(prices)))
    amix_inputs = len(prices) + 1  # 1 for main audio + N SFX
    filter_lines.append(f"[0:a]{sfx_labels}amix=inputs={amix_inputs}:normalize=0[aout]")
    # Reduce volume of all sounds audio by 30dB
    filter_lines.append(f"[aout]volume=-30dB[aout2]")
    
    print("filter_lines:", filter_lines)
    
    filter_complex = ";".join(filter_lines)
    
    return cmd, filter_complex

def overlay_text_and_audio(input_video, timestamps, prices, output_path):
    # ------------------------------------------------------------------
    # 1.  Read the source-file color tags so we can copy them verbatim
    # ------------------------------------------------------------------
    meta       = ffprobe_color_metadata(input_video)
    pix_fmt_in = meta.get("pix_fmt", "yuv420p")
    trc        = meta.get("color_trc", "bt709")
    prim       = meta.get("color_primaries", "bt709")
    csp        = meta.get("colorspace", "bt709")
    rng        = meta.get("color_range", "tv")          # tv = limited, pc = full

    # 10-bit HDR (HLG/PQ) clips need x265 + a 10-bit pixel-format
    is_10bit   = "10" in pix_fmt_in or trc.startswith(("smpte2084", "arib-std-b67"))
    codec      = "libx265" if is_10bit else "libx264"
    pix_fmt    = "yuv420p10le" if is_10bit else "yuv420p"
    
    # video = VideoFileClip(input_video)
    # w, h = video.w, video.h
    text_timing = [ -0.5, 1.0, 
                    1.0,  1.0, 
                    1.0,  1.0, 
                    1.0,  1.0, 
                    1.0,  1.0, 
                    1.0,  1.0, 
                    1.0,  1.0, 
                    1.0,  1.0, 
                    1.0,  1.0, 
                    1.0,  4.0, 
                    4.0,  8.0,
    ]
    all_timestamps = [ timestamps[0], timestamps[0], 
                       timestamps[0], timestamps[1], 
                       timestamps[1], timestamps[2], 
                       timestamps[2], timestamps[3], 
                       timestamps[3], timestamps[4],
                       timestamps[4], timestamps[5],
                       timestamps[5], timestamps[6],
                       timestamps[6], timestamps[7],
                       timestamps[7], timestamps[8],
                       timestamps[8], timestamps[9],
                       timestamps[9], timestamps[9],
    ]
    text_clips  = create_text_clips(prices, text_timing, all_timestamps)
    # audio_clips = create_audio_clips(timestamps)

    # Composite audio
    # base_audio = video.audio
    # for ac in audio_clips:
    #     base_audio = base_audio.overlay(ac)
    # final = final_vid.set_audio(base_audio)

    # final.write_videofile(output_path, codec="libx264", audio_codec="aac")
    
    ### old command with just text overlay
    # cmd = [
    #     "ffmpeg", "-y",
    #     "-threads", "0", # 0 = auto-detect max
    #     "-i", input_video,
    #     "-vf", text_clips,                        # overlay the text
    #     "-c:v", codec,
    #     "-pix_fmt", pix_fmt,
    #     "-preset", "ultrafast",
    #     "-crf", "35", # decrease this for better quality later
    #     "-c:a", "copy",                     # copy audio bit-perfect
    #     # keep the original color tags verbatim
    #     "-colorspace",   csp,
    #     "-color_primaries", prim,
    #     "-color_trc",    trc,
    #     "-color_range",  rng,
    #     output_path
    # ]
    
    cmd, audio_filter_complex = create_audio_clips_command(input_video, prices, all_timestamps, text_timing)

    # Build rest of command
    cmd += [
        "-vf", text_clips,
        "-filter_complex", audio_filter_complex,
        "-map", "0:v",          # video stream from main input
        "-map", "[aout2]",       # mixed audio
        "-c:v", codec,
        "-pix_fmt", pix_fmt,
        "-preset", "ultrafast",
        "-crf", "35",
        # "-c:a", "copy",   # removed this
        # Color tags as before
        "-colorspace", csp,
        "-color_primaries", prim,
        "-color_trc", trc,
        "-color_range", rng,
        output_path
    ]
    subprocess.run(cmd, check=True)

def main():
    args = parse_arguments()
    print(f"Started editing Day {args.day} - Set: {args.set}, Cost: ${args.cost:.2f}\n")
    
    raw_mov = args.video
    raw_mp4 = raw_mov.replace('.MOV', '.mp4')
    print(f"Started converting .MOV to .mp4")
    convert_mov_to_mp4(raw_mov, raw_mp4)
    print(f"Finished converting .MOV to .mp4\n")
    
    print(f"Started detecting card reveals.")
    timestamps = detect_card_reveals(raw_mp4, debug=False) # turn on debug frame here
    if len(timestamps) != MAX_REVEALS:
        print(f"ERROR: Expected {MAX_REVEALS} reveals, but detected {len(timestamps)} instead.")
        return
    print(f"Timestamps: {timestamps}")
    print("Finished detecting card reveals.\n")
    
    
    
    # Overlay text and sound
    prices_list = [float(p) for p in args.prices.split(',')]
    if len(prices_list) != 11:
        raise ValueError("Provide exactly 11 comma-separated prices.")
    
    output_path = raw_mp4.replace("raw_vids","edited_vids")
    print("Started overlaying text & sound.")
    overlay_text_and_audio(raw_mp4, timestamps, prices_list, output_path)
    print("Finished overlaying text & sound.")
    
    print(f"\nFinished editing Day {args.day}.")

if __name__ == '__main__':
    main()
    
    # # Print timestamps and show images of each detected card reveal:
    # print(f"Evaluate detected card reveals at the following timestamps:")
    # for ts in timestamps:
    #     print(f"Timestamp: {ts:.2f} seconds")
    #     frame = extract_frame(args.video, ts)
    #     if frame is not None:
    #         cv2.imshow(f"Frame at {ts:.2f}s", frame)
    #         cv2.waitKey(0)  # Wait for a key press to proceed to next frame
    #         cv2.destroyWindow(f"Frame at {ts:.2f}s")
    #     else:
    #         print("Could not extract frame at this timestamp.")
