import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

class dataBilgi():
    
    #dosyanın okunması
    def __init__(self,dosya):
        self.data = pd.read_csv(dosya)
        
    #NaN değerlerin temizlenmesi
    def temizle(self):
        self.data.dropna(how='any')
    
    #data ile ilgili bilgi
    def bilgi(self):
        self.data.info()

    def tanımla(self):
        self.data.describe()

        
class karsılastırma():
    
    #crosstab tablosunun index ve column olarak sunulması ve index isimlerin atanması
    def tablo(index, columns,normalize,  *args):
        index_name=[]
        tab = pd.crosstab(index=index, columns=columns, normalize=normalize)
        for arg in args:
            index_name.append(arg)
        tab.index=index_name
        return tab
    
    #tablo üzerinde istenilen duruma göre sıralama yapılması
    def sıralama():
        pass

class grafikle():
    
    def __init__(self):
        print(self)
    
    #datanın factorplot olarak görselleştrilmesi
    def factorPlot(column1, column2, hue, data):
        sns.factorplot(column1, column2, hue=hue, data=data)
        plt.show()
        
    #datanın barplot olarak görselleştirilmesi
    def barPlot(x, y, hue, data):
        sns.barplot(x=x, y=y, hue=hue, data=data)
        plt.show()
    
    #alt tablolamaların yapılması
    def subPlots():
        pass
    
    #datanı histogram olarak görselleştirilmesi
    def histogram():
        pass

        
class onisleme():
    
    def __init__(self):
        print(self)
        
    #datanın NaN değerlerden temizlenmesi
    def temizle(self):
        self.data.dropna(how='any')
        
    #datanın gerekli isterlere göre düzenlenmesi
    def düzenle(self):
        pass
    
    #tablolar için etiketlemelerin yapılması xlabel, ylabel vs.
    def etiketleme():
        pass
    
    #tekrar eden gereksiz değerlerin temizlenmesi
    def tekraredenTemizle(self,data):
        data.drop_duplicates(keep=False,inplace=True) 
    
    #datanın kolon bazlı birleştirilmesi
    #how:{‘left’, ‘right’, ‘outer’, ‘inner’} 
    def birlestirme(pd_1, pd_2, how, left_on, right_on):
        return pd_1.merge(pd_2, how=how, left_on=left_on, right_on=right_on)
  
    #datanın düzenlenmesi için ayrıntılı fonksiyonlar için
    def stratejiler():
        pass
