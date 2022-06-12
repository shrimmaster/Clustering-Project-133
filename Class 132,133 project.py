import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("star_with_gravity.csv")
df.head()

mass = df["Mass"].to_list()
radius = df["Radius"].to_list()
dist = df["Distance"].to_list()
gravity = df["Gravity"].to_list()
mass.sort()
radius.sort()
gravity.sort()
plt.plot(radius,mass)
#plt.plot(radius,gravity)

plt.title("Radius & Mass of the Star")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

plt.plot(mass,gravity)

plt.title("Mass vs Gravity")
plt.xlabel("Mass")
plt.ylabel("Gravity")
plt.show()

plt.scatter(radius,mass)
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

from wsgiref import headers
import pandas as pd
import matplotlib.pyplot as plt


print(headers)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns 

x=[]
for index,planet_mass in enumerate(mass):
  temp_list=[
             radius[index],
             planet_mass
             ]
  x.append(temp_list)

wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(x)
  wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),wcss,marker='o',color='red')

plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()