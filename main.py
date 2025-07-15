from utils import read_video, save_video
from trackers import Tracker

def main():
    video_frames = read_video("input_video/15sec_input_720p.mp4")
    tracker = Tracker("model/best (1).pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path='tracker_stubs/player_detection.pkl')
    output_video_frames = tracker.draw_annotations(video_frames, tracks)
    save_video(output_video_frames, 'output_videos/output.avi')

if __name__ == "__main__":
    main()
