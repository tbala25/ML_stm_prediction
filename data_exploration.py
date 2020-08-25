#######################################

from localLibrary_AWSConnector import *

def get_data_s3():
    data_list = []

    # Print out bucket names
    for bucket in s3.buckets.all():
        print(bucket.name + " contents: ")
        for obj in s3.Bucket(bucket.name).objects.all():
            print(obj.key)
            if('data/' in obj.key and obj.key != 'data/'):
                data_list.append(obj.key)

    return data_list

print(get_data_s3())


# # Load csv file directly into python
# obj = s3.Bucket('cheez-willikers').Object('foo.csv').get()
# foo = pd.read_csv(obj['Body'], index_col=0)
