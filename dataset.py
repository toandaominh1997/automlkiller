import os
from autotonne.utils import CloudService

with open(os.path.join(os.getcwd(), 'query.sql')) as fp:
    query_string = fp.read()
cloud = CloudService(project = 'vinid-data-science-prod')

df = cloud.read_gbq(query_string)
df.to_csv('data.csv')

