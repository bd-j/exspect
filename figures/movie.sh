framerate=24             # frames/second
frames_per_image=20       # separate images per second = frame_rate/frames_per_image

outname=paperfigures/nbands_movie.mp4
oroot=paperfigures/movie/aniband
rm ${oroot}*png


count=00
for im in paperfigures/nband?.png; do
  echo $im
  for ((n=0; n<=frames_per_image; n++)); do
    ((count=count+1));
    printf -v j "%02d" $count
    cp $im ${oroot}${j}.png;
    #echo $count
  done
done

rm $outname
ffmpeg -r $framerate -i ${oroot}%02d.png -c:v libx264 -pix_fmt yuv420p -y $outname
