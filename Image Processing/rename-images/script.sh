for i in ./jpeg/*.jpeg ; do convert "$i" "${i%.*}.jpg" ; done
