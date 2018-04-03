# Generate train.list and val.list from mat provided by:
# http://vision.stanford.edu/aditya86/ImageNetDogs/

from scipy import io

def main():
    trainmat = io.loadmat('/book/working/data/train_list.mat')
    f = open('/book/working/data/train.list', 'w')
    f.close()
    with open('/book/working/data/train.list', 'a') as f:
        for i, fs in enumerate(trainmat['file_list']):
            f.write("/book/working/data/Images/" + fs[0][0] + "\t" + str(trainmat['labels'][i][0]) + "\n")

    testmat = io.loadmat('/book/working/data/test_list.mat')
    f = open('/book/working/data/val.list', 'w')
    f.close()
    with open('/book/working/data/val.list', 'a') as f:
        for i, fs in enumerate(testmat['file_list']):
            f.write("/book/working/data/Images/" + fs[0][0] + "\t" + str(testmat['labels'][i][0]) + "\n")

if __name__ == "__main__":
    main()

