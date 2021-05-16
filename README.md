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
python .\xtarget.py --cam --camid 0
```

It starts into the mode `intro` which detects glare. If there is glare, it will be 
displayed in red on the screen. If you have glare either: 
* Make glare go away by using curtains, move target or camera position etc. 
* if glare is in target area
  * Press `j`/`k` to increase/decrease threshold
  * The less threshold you have, the less glare, but also the less likely that the laser will be detected
* if glare is outside target area
  * press `c` to crop the image

Once there is no glare, press `m` to go into `main` mode, where hits are detected. 
If you have glare, they will count as hits.

Optional:
* Press `t` to create a target circle (center first. Pts 0-100 based on radius)


## Troubleshooting

* hits do not get detected
  * change lightning
  * adjust threshold
  * change target material (use bright, reflective paper)
  * change distance and angle of the camera, or you
