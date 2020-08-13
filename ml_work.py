import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
#load white wine data
white_wine=pd.read_csv("winequality-white.csv",sep=";")

# standard scaler
scaler=StandardScaler()
#X,Y
X=white_wine.iloc[:,:-1]
X=scaler.fit_transform(X)
Y=white_wine.iloc[:,-1]
#train dataset, test dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#a. continious learning
#a.1 linear regression
reg = LinearRegression().fit(X_train, y_train)
#R square
y_hat=reg.predict(X_test)
plt.scatter(X_test[:,1],y_hat,c="r")
plt.scatter(X_test[:,1],y_test,c="b")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend(['predicted y','true y'])
plt.savefig("linear_regres.png")
#residuals od linear regression
lr_resid=(y_test-y_hat)**2
#a.2 random forest
rf_reg=RandomForestRegressor().fit(X_train,y_train)
y_hat_rf=rf_reg.predict(X_test)
plt.scatter(X_test[:,1],y_hat_rf,c="r")
plt.scatter(X_test[:,1],y_test,c="b")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Random Forest Regression')
plt.legend(['predicted y','true y'])
plt.savefig("random_forest.png")
#residuals of random forest
rf_resid=(y_test-y_hat_rf)**2
# t test for independent for residuals of linear regression and random forest
t_val,p_val=ttest_ind(lr_resid,rf_resid)
#########################################################

#b. clustering

#b.1. K-means
sill_km=np.zeros(30)
k = 11
for i in range(30):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_train)
    y_pred=kmeans.predict(X_test)
    sill_km[i]=silhouette_score(X_test, y_pred)

plt.scatter(X_test[:,1],y_pred,c=y_pred)
plt.xlabel('X')
plt.ylabel('Predicted Y')
plt.title('K-means clustering')
plt.savefig("k_means.png")

#b.2 GMM
sill_gm=np.zeros(30)
for i in range(30):
    gm = GaussianMixture(n_components=11)
    gm.fit(X_train)
    y_pred_gmm=gm.predict(X_test)
    sill_gm[i]=silhouette_score(X_test, y_pred_gmm)
        
plt.scatter(X_test[:,1],y_pred_gmm,c=y_pred_gmm)
plt.xlabel('X')
plt.ylabel('Predicted Y')
plt.title('GMM Clustering')
plt.savefig('GMM.png')
# t test for silhouette score of 30 different iteration of each clustering model
t_val_cluster,p_val_cluster=ttest_ind(sill_km,sill_gm)
#######################################################################

#c. Dimensionality reduction
#c.1. LLE
lle = LocallyLinearEmbedding(n_components=3)
lle.fit(X_train)
X_reduced_lle = lle.transform(X_test)
reconstruct_lle=lle.reconstruction_error_
color=y_test.values


plt.scatter(X_reduced_lle[:,1],X_reduced_lle[:,2],c=color)
plt.xlabel('First dim')
plt.ylabel('Second dim')
plt.title('LLE')
plt.savefig('lle.png')

#c.2.PCA
pca = PCA(n_components=3)
pca.fit(X_train)
X_reduced_pca = pca.transform(X_test)
X_reconstruct_pca=pca.inverse_transform(X_reduced_pca)
mse_pca=(X_test-X_reconstruct_pca)**2
sample_mean=np.mean(mse_pca,1)
#sort of equivalent to reconstruction error
total_mean=np.mean(sample_mean)


plt.scatter(X_reduced_pca[:,1],X_reduced_pca[:,2],c=color)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA')
plt.savefig('pca.png')



