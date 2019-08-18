#CODE-1---------------------------------------------------------------------------------------------

from sagemaker import get_execution_role

role = get_execution_role()
bucket = 'bucket-name' # Use the name of your s3 bucket here

#CODE-2---------------------------------------------------------------------------------------------

%%time
import pickle, gzip, numpy, urllib.request, json

# Load the dataset
urllib.request.urlretrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz","mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
	
#CODE-3----------------------------------------------------------------------------------------------

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (2,10)
def show_digit(img, caption='', subplot=None):
    if subplot == None:
        _, (subplot) = plt.subplots(1,1)
    imgr = img.reshape((28,28))
    subplot.axis('off')
    subplot.imshow(imgr, cmap='gray')
    plt.title(caption)
    
show_digit(train_set[0][30], 'This is a {}'.format(train_set[1][30]))

#If you use import numpy, all sub-modules and functions in the numpy module can only be accesses in the numpy.* namespace. For example numpy.array([1,2,3]).

#If you use import numpy as np, an alias for the namespace will be created. For example np.array([1,2,3]).

#If you use from numpy import *, all functions will be loaded into the local namespace. For example array([1,2,3]) can then be used.


#CODE-4------------------------------------------------------------------------------------------------

from sagemaker import KMeans

data_location = 's3://{}/kmeans_highlevel_example/data'.format(bucket)
output_location = 's3://{}/kmeans_highlevel_example/output'.format(bucket)

print('training data will be uploaded to: {}'.format(data_location))
print('training artifacts will be uploaded to: {}'.format(output_location))

kmeans = KMeans(role=role,
                train_instance_count=2,
                train_instance_type='ml.c4.8xlarge',
                output_path=output_location,
                k=10,
                data_location=data_location)


#CODE-5---------------------------------------------------------------------------------------------

%%time

kmeans.fit(kmeans.record_set(train_set[0]))


#CODE-6---------------------------------------------------------------------------------------------

%%time

kmeans_predictor = kmeans.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')


#CODE-7---------------------------------------------------------------------------------------------

import sagemaker
from time import gmtime, strftime

job_name = 'Batch-Transform-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
prefix = 'sagemaker/project_name'

# Initialize the transformer object
transformer =sagemaker.transformer.Transformer(
     model_name=model_name,
     instance_count=1,
     instance_type='ml.c4.xlarge',
     output_path="s3://{}/{}/Batch-Transform/".format(bucket,prefix)
)

# To start a transform job:
sample_data_bucket = sample-data-bucket.format(region)
input_file_path = 'path-to-your-samples'
transformer.transform('s3://{}/{}'.format(sample_data_bucket, input_file_path),content_type='text/csv')
# Them wait until transform job is completed
transformer.wait()

# To stop a transform job:
transformer.stop_transform_job()



#CODE-8---------------------------------------------------------------------------------------------

result = kmeans_predictor.predict(valid_set[0][30:31])
print(result)


#CODE-9---------------------------------------------------------------------------------------------

for cluster in range(10):
    print('\n\n\nCluster {}:'.format(int(cluster)))
    digits = [ img for l, img in zip(clusters, valid_set[0]) if int(l) == cluster ]
    height = ((len(digits)-1)//5) + 1
    width = 5
    plt.rcParams["figure.figsize"] = (width,height)
    _, subplots = plt.subplots(height, width)
    subplots = numpy.ndarray.flatten(subplots)
    for subplot, image in zip(subplots, digits):
        show_digit(image, subplot=subplot)
    for subplot in subplots[len(digits):]:
        subplot.axis('off')
		
    plt.show()

#CODE-10---------------------------------------------------------------------------------------------

#CLEAN UP

print(kmeans_predictor.endpoint)

import sagemaker
sagemaker.Session().delete_endpoint(kmeans_predictor.endpoint)










