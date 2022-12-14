# orange_peel_analysis

### 本專案主要透過影像處理以及機器學習等方法進行車漆的品質檢測
##### 透過正規化等方法進行影像前處理，來抓取到標準化後的邊緣影像，接著透過紋理分析的方法進行影像特徵提取，之後再針對提取出來的特徵進行預測。

## 實作流程
+ orange_peel_128.py
+ texture_analysis.py 
+ model_predict.py

## orange_peel_128
#### 將原始影像轉換成標準化後的影像
##### 為了要使後續紋理特徵提取富有意義，需要在前處裡的過程中讓所有燈管的成像達到統一，因此透過**orange_peel_128**這隻程式碼將車廠實際拍攝影像進行轉換，達到標準化的過程。

## texture_analysis
#### 將標準化後的影像提取其中的紋理特徵
##### 為了分析車漆品質的好壞，希望透過物理學原理中的偏轉效應來判斷車漆的好壞，因此在此採取紋理特徵的方式將影像中的紋理細節提取出來。
##### 使用
+ GLCM
+ GLRLM
+ GLDM
+ NGTDM
+ GLSZM

## model_predict
#### 將紋理特徵透過機器學習進行預測
##### 為了預測車漆品質，透過Adaboost的模型將代表車漆品質的四個值分別進行預測。
##### 車漆品質包含
+ NID
+ S
+ L
+ R
