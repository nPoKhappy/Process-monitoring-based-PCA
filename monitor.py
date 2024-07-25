# -*- coding: utf-8 -*-
"""
PCA and KPCA 演算法
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif']    = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
np.set_printoptions(precision=3,suppress=True)

#before use the class,you have to normalize the data first using standardscaler module import from sklearn.preprocesing


class PCA():
    #construction func
    def __init__(self , training_data : np.ndarray  , testing_data , number_sample,draw , lv中的係數貢獻 , variance_attribution_cutoff=95):
       #fit and transform training data and testing data
        trainingdata_scaled = sc.fit_transform(training_data)
        if testing_data is not None:
            testingdata_scaled = sc.transform(testing_data)
        
       
        #initiate the variable
        self.trainingdata_scaled = trainingdata_scaled
        self.testingdata_scaled = testingdata_scaled if testing_data is not None else None
        self.number_sample = number_sample
        self.draw = draw
        self.lv中的係數貢獻 = lv中的係數貢獻
        self.variance_attribution_cutoff = variance_attribution_cutoff
        
        #these attributes down below can store the computed values and be callable if needed
        self.Q_tr = None
        self.T2_tr = None
        self.Q_ts = None
        self.T2_ts = None
        self.T2lim = None
        self.Qlim = None
        self.residue = None
        self.score_tr = None
        self.P = None
        self.lv = None
        
    def perform(self):
        # Initialize to prevent not be able to access local variable
        Q_tr = 0
        Qlim = 0
        residue = 0
        #getting training data covariance and use SVD to get eigenvalue and eigenvetor
        cov_training = np.cov(self.trainingdata_scaled.T)
        U,S,VT = np.linalg.svd(cov_training)
       
        #Getting variance from cumsum eigenvalue
        variance = S*100 / np.sum(S)
        cumsum_variance = np.cumsum(variance)

        #lv 也就是大於95%選定的PC 數量
        lv = int(np.where(cumsum_variance > self.variance_attribution_cutoff)[0][0]+1)
        
        #選定前幾個PCs
        P=U[:,:lv]
        #計算score (也就是投影點)
        score_tr=np.dot(self.trainingdata_scaled,P)
        
        draw_variance = 0
        if draw_variance ==1:
            print("      percent variance captured by PCs      ")
            print("--------------------------------------------")
            print('Principal     Eigenvalue     % Variance     % Variance')
            print('Component         of          Captured       Captured')
            print(' Number         Cov(X)        This  PC        Total')
            print('---------     ----------     ----------     ----------')
    
            sf=" {}           {:2e}         {:>.2f}           {:.2f} "

            for i in range(lv):
                formatted_string=sf.format(i+1,S[i],variance[i],cumsum_variance[i]) 
                print(formatted_string)
            
        #calculate the t2 value
        t2 = np.sum(score_tr**2/(S[:lv]) ,axis=1)
        
        
        #compute the t2lim value
        from scipy.stats import f

        term_first = lv*(self.number_sample-1)*(self.number_sample+1)/(self.number_sample*(self.number_sample-lv))
        T2lim_tr = term_first * f.ppf(q=1-0.05,dfn=lv,dfd=min(self.trainingdata_scaled.size,self.number_sample-lv))
        
        
        #draw t2 and t2lim value

        if self.draw == 1:
            plt.figure(dpi=150)
            plt.plot(t2,'b',
                     t2,'+g',
                     [0,self.number_sample],[T2lim_tr,T2lim_tr],'--r')
            plt.ylim([0,T2lim_tr*5])
            plt.grid()
            plt.title("T訓練")
            plt.show()
        
        
        
        if lv < 3:
            #compute q value
            #[num_sample,num_variables]     #dot([num_samples,projected_numbers],[num_variables,num_projected_numbers]^T)
            residue = self.trainingdata_scaled-np.dot(score_tr,P.T) #Residue
            Q_tr = np.sum(residue**2,axis = 1)
            
            #compute qlim
            lamda = S[lv:]
            theta1 = np.sum(lamda**1)
            theta2 = np.sum(lamda**2)
            theta3 = np.sum(lamda**3)
            
            h0 = 1-2*theta1*theta3/3/theta2**2
            c_alpha = 1.65
            
            term1 = c_alpha*np.sqrt(2*theta2*h0**2)/theta1
            term2 = theta2*h0*(h0-1)/theta1**2
            
            Qlim = theta1*((term1+1+term2)**(1/h0))

            if h0<0.0:
                h0=0.0001
                print("Warning:distribution unused eigenvalues indicates")
                print("        that you should probably retain more PCs in the model.")
        
        #draw q and qlim value

        if self.draw== 1:
            plt.figure(dpi=150)
            plt.plot(Q_tr,"b",
                     Q_tr,"+g",
                     [0,self.number_sample],[Qlim,Qlim],'--r')
            
            plt.grid()
            plt.title("Q訓練")
            plt.ylim([0, 5*Qlim])
            plt.show()
        
        
        
        #查看原變量在lv貢獻
        if self.lv中的係數貢獻 == 1:
            for i in range(lv):
                plt.figure(figsize=[6,6])
                plt.title("lv"+str(i+1)+'中對應原變量的係數')
                plt.bar(np.linspace(1,len(P[:,i]),len(P[:,i])).astype(int).astype(str), abs(P[:,i]))
                plt.show()
        
        #if testing_data is not none
        
        if self.testingdata_scaled is not None:
            #標準化後的數據計算score
            score_ts = np.dot(self.testingdata_scaled,P)
    
            #T2值計算
    
            T2_ts = np.sum(score_ts**2/S[:lv] , axis=1)
    
    
        
            #畫T2圖
            if self.draw ==1:
                plt.figure(dpi=150)
                plt.plot(T2_ts,'-r',
                         T2_ts,'+g',
                         [0,self.number_sample],[T2lim_tr,T2lim_tr],'--b'),
                
                         
                plt.grid();plt.legend();plt.title("T測試");plt.show()
                
            if Qlim != 0:
                #Q計算
                if self.draw == 1:
                    Q_ts = self.testingdata_scaled - np.dot(score_ts,P.T) #residue
                    Q_ts_sum = np.sum(Q_ts**2,axis=1)
                    #Q畫圖
                    plt.figure(dpi=150)
                    plt.plot(Q_ts_sum,"-r",
                         Q_ts_sum,"+g",
                        [0,self.number_sample],[Qlim,Qlim],'--b')
    
                    plt.grid();plt.legend();plt.title("Q測試");plt.show()  
            #儲存各個參數、值以便傳喚
            self.Q_ts = Q_ts
            self.T2_ts = T2_ts
        #Assigning parameter 
        self.Q_tr = Q_tr
        self.T2_tr = t2
        self.T2lim = T2lim_tr
        self.Qlim = Qlim
        self.residue = residue
        self.score_tr = score_tr
        self.P = P
        self.lv = lv
    #reconstruct linear data back to original space
    def reconstruction(self,original_data,draw_=0):  #先指定draw_==0
        
        #obtain the mean and standard deviation from standardscaler module
        mean = sc.mean_
        std = sc.scale_
        
        #recontruct the pca model data
        x_hat = np.dot(self.score_tr,self.P.T) #[num_samples,num_variables] 
        
        #data mutiplied by std and added mean 
        reconstruction = x_hat * std + mean
        
        #distance for each variable in each sample
        self.distance = np.abs((original_data - reconstruction))
        #each variable in one sample distace adding up 
        self.sample_dis = np.sum(self.distance,axis=1)
        
       
        
        #draw the original and reconstruction data
        if draw_ == 1:
            test = plt.figure(dpi=150)
            ax_test = test.add_subplot(111,projection="3d")
            #reconstruction 
            ax_test.plot(reconstruction[:,0],reconstruction[:,1],reconstruction[:,2],color="red",label="reconstruction")
            #original data
            ax_test.plot(original_data[:,0],original_data[:,1],original_data[:,2],color="blue",label="original")
            
            ax_test.plot(self.trainingdata_scaled[:,0],self.trainingdata_scaled[:,1],self.trainingdata_scaled[:,2],label="tangent",color="black")
   
            ax_test.set_title("original and tanget_line and reconstruction line")
            ax_test.legend()
        
        return self.sample_dis,self.distance
    
    #check if the data is standardized
    def check_standardized(self,data):
        #if data is none,not 2d,and nan,raise warning
        if data is None:
            raise ValueError("Input data is None")
        
        if data.ndim != 2:
            raise ValueError("Expected 2D array, got {}D array instead".format(data.ndim))
    
        # Check for NaN values
        if np.isnan(data).any():
            raise ValueError("Input data contains NaN values")
            
        #check if the data is normalize
        sc.fit(data)
        #obtain the data mean and standard deviation
        means = sc.mean_
        stds = np.sqrt(sc.var_)
        
        tolerance = 1e-5
        
        is_standardized = np.all(np.abs(means) < tolerance) and np.all(np.abs(stds - 1) < tolerance)
        
        if not is_standardized:
            return False
        else:
            return True
        


        


        
        
def KDE(data,bw=None,alpha=0.95): ####利用KDE計算累積概率為alpha的對應X值
    #### 參考網址 https://zhuanlan.zhihu.com/p/36859670
    ## 先用sklearn的KDE擬合
    ## 首先將數據尺度縮放到近似無窮大，然後根據近似微分求解
    data=data.reshape(-1,1)
    Min=np.min(data)
    Max=np.max(data)
    Range=Max-Min
    ## 起點和終點
    x_start=Min-Range
    x_end=Max+Range
    ### nums越大之後估計的累積概率越大
    nums=2**12
    dx=(x_end-x_start)/(nums-1)
    data_plot=np.linspace(x_start,x_end,nums)
    if bw is None:
        ##最佳帶寬選擇
        ##參考：Adrian W, Bowman Adelchi Azzalini
        # - Applied Smoothing Techniques for Data Analysis_
        # The Kernel Approach with S-Plus Illustrations (1997)
        ##章節2.4.2 Normal optimal smoothing,中位數估計方差效果更好，
        #與matlab的ksdensity一致
        data_median=np.median(data)
        new_median=np.median(np.abs(data-data_median))/0.6745
        ##np.std(data,ddof=1)當ddof=1時計算無偏標準差，即除以n-1，為0時除以n
        bw=new_median*((4/(3*data.shape[0]))**0.2)
        if bw <= 0:
            bw = 0.001

#導入核函數
from sklearn.metrics.pairwise import rbf_kernel 
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

        
class KPCA():
    """
    training_data,testing_data must be scaled at first
    gamma is the rbf kernel parameter
    draw is to set 0 or 1 to draw the T2 and Q statistics 
    lv is the chosen dimension to project the data from feature space
    kernel function is provide within rbf,sigmoid,linear and cosine
    
    """
    def __init__(self,trainingdata_scaled,testingdata_scaled,gamma,draw,lv,Num_sample,kernelfucntion):
        self.trainingdata = trainingdata_scaled
        self.testingdata = testingdata_scaled
        self.gamme = gamma
        self.draw = draw
        self.lv = lv
        self.N = Num_sample
        self.kernel = kernelfucntion
        
        
    def kpca_perform(self):
        if self.kernel == 'rbf':
            K_tr = rbf_kernel(self.trainingdata, gamma=self.gamma)
        elif self.kernel == 'sigmoid':
            K_tr = sigmoid_kernel(self.trainingdata, gamma=self.gamma)
        elif self.kernel == 'linear':
            K_tr = linear_kernel(self.trainingdata)
        elif self.kernel == 'cosine':
            K_tr = cosine_similarity(self.trainingdata, self.trainingdata)
        else:
            raise ValueError("Invalid kernel function specified. Please choose from 'rbf', 'sigmoid', 'linear', or 'cosine'.")
        K_tr = self.kernel(self.trainingdata,self.trainingdata)

        #中心化
        shape = np.shape(K_tr)
        one_tr = np.ones(shape)/shape[0]
        K_tr = K_tr - one_tr.dot(K_tr) - K_tr.dot(one_tr) + one_tr.dot(K_tr).dot(one_tr)

        #取特徵值 特徵向量
        from scipy.linalg import eigh
        S, A = eigh(K_tr)



        # 選前幾個投影PCs
        U = np.column_stack((A[:, -i] for i in range(1, self.lv + 1)))
        V = np.column_stack((A[:, -i] for i in range(1, self.N+1)))

        lambdas_R  = [S[-i] for i in range(1, self.lv+1)]
        lambdas  = [S[-i] for i in range(1, self.N+1)]

        #標準化投影向量
        U = U / np.sqrt(lambdas_R)
        V = V / np.sqrt(lambdas)

        #計算投影後的數據
        x_score_tr_R = K_tr.dot(U) 
        x_score_tr = np.nan_to_num(K_tr.dot(V) )

        # 監控
        
        T2_tr = np.sum( (x_score_tr_R)**2/lambdas_R, axis = 1 )
        T2lim = KDE(T2_tr)

        Q_tr =  np.sum(x_score_tr**2,axis = 1) - np.sum(x_score_tr_R**2,axis=1) 
        Qlim = KDE(Q_tr)
        #監控畫圖
        if self.draw == 1:
            plt.figure(dpi=150)
            plt.subplot(211)
            plt.title("T2 訓練")
            plt.plot(T2_tr[:],'-r',
                     T2_tr[:],'+g',
                     [0,self.N],[T2lim,T2lim],'--b')
            plt.ylim([0,T2lim*5]);plt.grid()
            
            
            plt.subplot(212)
            plt.title("Q 訓練")
            plt.plot(Q_tr[:],'-r',
                     Q_tr[:],'+g',
                     [0,self.N],[Qlim,Qlim],'--b')
            plt.ylim([0,Qlim*5]);plt.grid();plt.show()
       
        
        if self.testingdata is not None:
           K_ts = self.kernel(self.trainingdata,self.trainingdata)


           #1矩陣
           N_ts = np.shape(K_ts)[0]
           one_ts = np.ones([N_ts,N_ts])/N_ts

           #對核矩陣進行中心化
           K_ts = K_ts - one_ts.dot(K_tr) - K_ts.dot(one_tr) + one_ts.dot(K_tr).dot(one_tr)

           #用前面的標準化後的PCs
           x_score_ts_R = K_ts.dot(U)
           x_score_ts = np.nan_to_num( K_ts.dot(V) )

           #監控
           T2_ts = np.sum(x_score_ts_R**2/lambdas_R,axis=1)

           Q_ts = np.sum(x_score_ts**2,axis=1) - np.sum(x_score_ts_R**2,axis=1)


           if self.draw == 1:
               #T2 測試
               plt.figure(dpi=150)
               plt.subplot(211)
               plt.plot(T2_ts[:],'-r',
                        T2_ts[:],'+g',
                        [0,self.N],[T2lim,T2lim],'--b')
               plt.title("T2 測試")
               plt.grid();plt.ylim([0,5*T2lim])
               
               #Q 測試
               plt.subplot(212)
               plt.plot(Q_ts[:],'-r',
                        Q_ts[:],'+g',
                        [0,self.N],[Qlim,Qlim],'--b')
               plt.ylim([0,5*Qlim]);plt.grid();plt.title("Q測試")
               plt.show()
        
        
        def plot(self):
            """原數據畫圖,3維"""
            fig1 = plt.figure(figsize=[12,12],dpi=150)
            ax1 = fig1.add_subplot(221,projection="3d")
            plt.title("原尺度空間數據")
            ax1.scatter(self.training_data[:,0],self.trainingdata[:,1],self.trainingdata[:,2],color='r',
                     label="訓練數據")
            ax1.scatter(self.testingdata[:,0],self.testingdata[:,1],self.testingdata[:,2],color='y',
                     label="測試數據")
            ax1.view_init(elev=20,azim=80)

            plt.legend()
            plt.show()


            plt.figure(figsize=[12,12],dpi=150)
            plt.subplot(212)
            plt.title("投影空間數據")
        
       
        
        