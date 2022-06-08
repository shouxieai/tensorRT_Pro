#!/bin/bash

sopath=/lib/x86_64-linux-gnu

if [ ! -L ${sopath}/libcuda.so ]; then
    files=(`find $sopath/libcuda.so*`)
    raw_so=${files[0]}
    echo Create soft link ${raw_so}
    sudo ln -s ${raw_so} ${sopath}/libcuda.so
fi

if [ ! -L ${sopath}/libnvcuvid.so ]; then
    echo Create soft link ${sopath}/libnvcuvid.so.1
    sudo ln -s ${sopath}/libnvcuvid.so.1 ${sopath}/libnvcuvid.so
fi

if [ ! -L ${sopath}/libnvidia-encode.so ]; then
    echo Create soft link ${sopath}/libnvidia-encode.so.1
    sudo ln -s ${sopath}/libnvidia-encode.so.1 ${sopath}/libnvidia-encode.so
fi

# process ffmpeg soft link
files=`find ffmpeg/lib/lib*.so.*`
pwd=`pwd`

for file in $files
do
    dst=`echo $file | sed 's/.so.*/.so/'`
    dst=`echo $dst  | sed 's/-[0-9a-z]*.so/.so/'`
    if [ ! -L $dst ]; then
        echo Create ffmpeg soft link $dst
        ln -s $pwd/$file $pwd/$dst
    fi
done