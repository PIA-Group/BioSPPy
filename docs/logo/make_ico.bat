magick favicon.png -resize 16x16 fav_16.png
magick favicon.png -resize 31x31 fav_32.png
magick favicon.png -resize 64x64 fav_64.png
magick fav_16.png fav_32.png fav_64.png ../favicon.ico
magick identify ../favicon.ico
del fav_16.png fav_32.png fav_64.png
pause