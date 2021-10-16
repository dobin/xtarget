from gooey import Gooey, GooeyParser
from playback import Playback
from videostream import FileVideoStream, CamVideoStream
import argparse


@Gooey(show_success_modal=False, show_restart_button=False, disable_progress_bar_animation=True)
def main():
    ap = argparse.ArgumentParser()
    # ap = GooeyParser(description="settings_msg")
    ap.add_argument("--cam", help="Webcam ID", action='count', default=0)

    ap.add_argument("--enableTarget", help="Enable detection of iTarget or similar target", action='store_true')
    ap.add_argument("--camProjector", help="Use projector", action='store_true')
    ap.add_argument("--saveHits", help="Save picture of hits", action='store_true')
    args = ap.parse_args()

    if args.cam is not None:
        camId = int(args.cam)
        videoStream = CamVideoStream(threaded=True)

        resolution = {'width': 1920, 'height': 1080}
        #if args.width is not None and args.height is not None:
        #    resolution = {'width': args.width, 'height': args.height}

        videoStream.initCam(camId, resolution=resolution)
        playback = Playback(
            videoStream,
            withProjector=args.camProjector, saveHits=args.saveHits, enableTarget=args.enableTarget)
        playback.init()
        playback.play()


if __name__ == "__main__":
    main()
