# copy data into 'trans', compress it into 7z and send it with wormhole

path=$@

rm -rf trans
mkdir trans
cp -r $path ./trans
rm -f trans.7z

7zr a trans.7z ./trans

wormhole send trans.7z

