from ir_dataset_package.dataset import QrelTripletDataset



def test1():
    dataset = QrelTripletDataset(
        collection_path='/mnt/nfs/scratch1/jkillingback/ANCE_DATA/data/raw_data/collection.tsv',
        query_path='/mnt/nfs/scratch1/jkillingback/ANCE_DATA/data/raw_data/queries.train.tsv',
        qrels_path='/mnt/nfs/scratch1/jkillingback/ANCE_DATA/data/raw_data/qrels.train.tsv',
        cache_dir='/mnt/nfs/scratch1/jkillingback/cache/ir_dataset_2'
    )

    for i in range(10):
        print(dataset[i])

def main():
    test1()

if __name__ == '__main__':
    main()