curl -o /home/ubuntu/data/images.tar http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
curl -o /home/ubuntu/data/lists.tar http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar

tar -xvf /home/ubuntu/data/images.tar -C /home/ubuntu/data/
tar -xvf /home/ubuntu/data/lists.tar -C /home/ubuntu/data/

rm /home/ubuntu/data/images.tar
rm /home/ubuntu/data/lists.tar

