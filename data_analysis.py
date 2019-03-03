
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from matplotlib.dates import AutoDateLocator,DateFormatter,MonthLocator
from sklearn.datasets import load_iris
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn import metrics
#以下为确定degree所使用的sklearn库
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Perceptron
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split



def open_csv(path,filename):
	f = open(path + filename)#此种打开读取方式可以避免中文名称带来的文件读取错误
	df = pd.read_csv(f)
	return df 

def draw_single_confidance(df):
	plt.figure()
	x_date = pd.to_datetime(df["x"])#处理日期格式，使其能够进行被当作横坐标
	x = x_date
	y1 = df["fitted_values"]
	y2 = df["upper_bound"]
	y3 = df["lower_bound"]
	y4 = df["input_data.count"]

	# plt.xlabel(xaxis,fontproperties = 'SimHei',fontsize = 25)#设置坐标轴格式，下同
	# plt.ylabel(yaxis,fontproperties = 'SimHei',fontsize = 25)
	# plt.title(title,fontproperties = 'SimHei',fontsize = 25)#设置标题

	plt.xticks(rotation=45)
	plt.gca().yaxis.get_major_formatter().set_powerlimits((0,100000000000000)) 

	plt.plot(x,y1,color='black',linewidth=1.0,linestyle='-')
	plt.plot(x,y2,color='black',linewidth=0.5,linestyle='--')
	plt.plot(x,y3,color='black',linewidth=0.5,linestyle='--')
	for i in range(0,len(df['x'])):
		if df['output_boo'][i] == True:
			plt.plot_date(x[i],y4[i],fmt='b.')
		else:
			plt.plot_date(x[i],y4[i],fmt='r.')

	plt.show()

# def top_rmbplayer(df,t,l,r,d):
# 	df = removal(df)
# 	# df['ratio'] = (-df['total_lose'])/df['total_gain']
# 	df['ratio'] = (-df['消耗的虚拟货币总数量'])/df['购买虚拟货币总数量']
# 	df['unusual_point'] = df.apply(lambda x: unusual(x.在线总时长（精确到分钟）,x.最高等级,x.ratio,x.in_game_days,t,l,r,d),axis = 1)
# 	# df['unusual_point'] = df.apply(lambda x: unusual(x.duration,x.level,x.ratio,x.in_game_days,t,l,r,d),axis = 1)
# 	df.to_excel("top_rmbplayer_unusual_point.xlsx")

def unusual(time,level,ratio,days,t,l,r,d):
	unusual = []
	if time < 100 or time < t:
		unusual.append('在线时间过短,')
	if level < 10 :
		unusual.append('最高等级过低,')
	if ratio < 0.8 :
		unusual.append('充值消耗比过低,')
	if days < 3 or days < d:
		unusual.append('游戏内天数过短,')
	unusual = [str(n) for n in unusual]
	unusual = "".join(unusual)
	return unusual

def removal(df):
	# df = pd.pivot_table(df,index=["year","a.accountid"],values=["pay","total_gain","total_lose","totalcoin","duration","logintimes","level","in_game_days"],aggfunc=np.sum,fill_value=0) 
	df = pd.pivot_table(df,index=["年份","盖娅id"],values=["充值总金额（单位：元）","购买虚拟货币总数量","消耗的虚拟货币总数量","剩余虚拟币数量","在线总时长（精确到分钟）","登录次数","最高等级","in_game_days"],aggfunc=np.sum,fill_value=0) 
	return df

def fitting_function(df,degree):
	x = df[['date']]
	y = df[['count']]
	classifier = Pipeline([('poly', PolynomialFeatures(degree=degree)), ('linear', linear_model.RANSACRegressor ())])
	classifier.fit(x, y)  # 训练数据来学习，不需要返回值
	y_predict = classifier.predict(x)  # 测试数据，分类返回标记
	df['y_predict'] = y_predict
	s=np.sqrt(metrics.mean_squared_error(y_predict,y))
	df["upper_bound"] = y_predict + 2 * s/len(x)
	df["lower_bound"] = y_predict - 2 * s/len(x)
	#差一个上下限判断的条件
	return df

def get_degree(df):
	#此函数的目的是为了能够确定多项式拟合的最优阶数
	x = df[['date']]
	y = df[['count']]
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
	#将数据读取之后将数据分为训练集和测试集
	rmses = []
	r2_scores = []
	degrees = np.arange(1, 36)
	min_rmse, min_deg,score = 1e10, 0 ,0
	#设定几个指标的默认值
	for deg in degrees:
		# 生成多项式特征集(如根据degree=3 ,生成 [[x,x**2,x**3]] )
		poly = PolynomialFeatures(degree=deg, include_bias=False)
		x_train_poly = poly.fit_transform(x_train)
	 
		# 多项式拟合
		poly_reg = LinearRegression()
		poly_reg.fit(x_train_poly, y_train)
		#print(poly_reg.coef_,poly_reg.intercept_) #系数及常数
		
		# 测试集比较
		x_test_poly = poly.fit_transform(x_test)
		y_test_pred = poly_reg.predict(x_test_poly)
		
		#mean_squared_error(y_true, y_pred) #均方误差回归损失,越小越好。
		poly_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
		rmses.append(poly_rmse)
		# r2 范围[0，1]，R2越接近1拟合越好。
		r2score = r2_score(y_test, y_test_pred)
		
		# degree交叉验证
		if min_rmse > poly_rmse:
			min_rmse = poly_rmse
			min_deg = deg
			score = r2score
		print('degree = %s, RMSE = %.2f ,r2_score = %.2f' % (deg, poly_rmse,r2score))
		#RMSE 最小，R平方非负且接近于1，则模型最好
		r2_scores.append(r2score)

path = 'C:\\work file\\Rtool\\'
filename = 'number.csv'
df = open_csv(path,filename)
get_degree(df)



##拟合并选取置信区间
# path = 'C:\\work file\\Rtool\\'
# filename = 'number.csv'
# df = open_csv(path,filename)
# degree=7
# fitting_function(path,filename,degree)


##大R玩家处理
# path = 'C:\\work file\\Gaea\\D&A\\系统数据\\仙剑-台湾\\台湾仙剑(全)\\台湾仙剑\\大R玩家'
# filename = '表格示例5.csv'
# df = open_csv(path,filename)
# t=df.duration.mean()
# l=df.level.mean()
# r=10
# d=df.in_game_days.mean()
# top_rmbplayer(df,t,l,r,d)


# ##置信区间画图
# path = 'C:\\work file\\Rtool\\'
# filename = 'count.csv'
# df = open_csv(path,filename)
# draw_single_confidance(df)
