#!/bin/bash

ls /usr/local/libtorch/lib | grep -x ".*\.so" >> /usr/local/lib/libtorchlib.txt
cd /usr/local/lib

infile=libtorchlib.txt
outfile=includeLibraries.h

echo "#pragma cling load(\"/usr/local/lib/libtesseract.so."$1"\")" >> "$outfile"
echo "#pragma cling add_include_path(\"/usr/local/libtorch/include\")" >> "$outfile"
echo "#pragma cling add_include_path(\"/usr/local/libtorch/include/torch/csrc/api/include\")" >> "$outfile"
echo "#pragma cling add_library_path(\"/usr/local/libtorch/lib\")" >> "$outfile"
echo "#pragma cling load(\"/usr/local/lib/libdlib.so\")" >> "$outfile"

while read line
do
        echo "#pragma cling load(\"/usr/local/libtorch/lib/$line\")" >> "$outfile"
done < "$infile"

rm -f libtorchlib.txt
