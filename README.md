# xTarget

Identifying laser points on targets. 


## Install 

```
pip install -r requirements.txt
```

The webcam resolution will be set to 1920x1080 by default. Everything else is untested. 

## Basic usage

To use webcam with id `0`:
```
python .\xtarget.py --cam 0
```

It starts into the mode `intro` which detects glare. If there is glare, it will be 
displayed in red on the screen. If you have glare either: 
* Make glare go away by using curtains, move target or camera position etc. 
* if glare is in target area
  * Press `j`/`k` to increase/decrease threshold
  * The less threshold you have, the less glare, but also the less likely that the laser will be detected
* if glare is outside target area
  * press `c` to crop the image

Once there is no glare, press `[space]` to go into `main` mode, where hits are detected. 
If you have glare, they will count as hits.

Optional:
* Press `t` to create a target circle (center first. Pts 0-100 based on radius)


## Troubleshooting

* hits do not get detected
  * change lightning
  * adjust threshold
  * change target material (use bright, reflective paper)
  * change distance and angle of the camera, or you


## Other notes

* Hero4 gopro videos make opencv crash because of audio track. Remove audio with: 
  * `.\ffmpeg.exe -i .\test.mp4 -vcodec copy -an test_out.mp4`


## Keyboard Shortcuts

* c: crop (c to quit)
* t: crop (t to quit)
* q: quit
* p: pause
* j: increase thresh
* k: decrease thresh
* g: show glare
* space: change mode
* s: savehits 

* video playback:
    * d: go one frame back
    * f: go one frame forward
    * e: go 10 frames back
    * p: pause


## Other modes

### Display a frame of a video, for research purposes

```
python xtarget.py --showframe test.mp4 --framenr 123
```

### Tests

```
python xtarget.py --tests
```

### Tests quick

```
python xtarget.py --testsQuick
```

## Write video data for tests

```
python xtarget.py --write test.mp4
```

