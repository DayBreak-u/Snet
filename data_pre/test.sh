DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

echo "Downloading..."

wget -c http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

echo "Unzipping..."

tar -xf caffe_ilsvrc12.tar.gz && rm -f caffe_ilsvrc12.tar.gz

echo "Done."