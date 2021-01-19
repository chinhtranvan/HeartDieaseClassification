# HeartDieaseClassification
1	DATA. 

  Link: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
  
  
  This is database of patient about heart disease. This data was taken by University of Switzerland and V.A. Medical Center, Long Beach and Cleveland Clinic Foundation, Hungarian Institute of Cardiology, Budapest.Each of them have different number of samples. Cleveland:303, Hungarian:294,Switzerland:123, and long beach VA:200. All attributes are numeric value. Each database has the same instance format. This databases have 76 features, all published experiments refer to using a subset of 14 of them (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak , slope, ca, thal, diagnosis). The output is presence of heart disease in the patient from 0 to 4 (5 outputs):Coronary artery (atherosclerotic) heart disease that affects the arteries to the heart ( value 4), Valvular heart disease that affects how the valves function to regulate blood flow in and out of the hear (value 3), Cardiomyopathy that affects how the heart muscle squeezes (value 2), Heart rhythm disturbances (arrhythmias) that affect the electrical conduction (value 1), Absence of heart disease (value 0).

2 DESCRIBING FEATURE



<img width="378" alt="Capture" src="https://user-images.githubusercontent.com/47764275/104999612-235ea280-59fb-11eb-958f-b15b5ca98202.PNG">
3

PREPROCESSING

![aaa](https://user-images.githubusercontent.com/47764275/105008447-3d9e7d80-5a07-11eb-9a33-ba0d84e6a091.png)
Repace ? by Nanvalue 
![bbb](https://user-images.githubusercontent.com/47764275/105008486-4c853000-5a07-11eb-97b1-49cccfa5c840.png)
Remove Nan value
![ggg](https://user-images.githubusercontent.com/47764275/105008933-e056fc00-5a07-11eb-9c3a-3e7dc24329dd.png)
Normalization
![ttt](https://user-images.githubusercontent.com/47764275/105009019-f95fad00-5a07-11eb-89fd-576cc3ed72ef.png)
Result

![ccc](https://user-images.githubusercontent.com/47764275/105009121-1dbb8980-5a08-11eb-87aa-ae99c5167c62.png)
Random forest curve close to the perfect ROC curve have a better performance level than the ones
![ddd](https://user-images.githubusercontent.com/47764275/105009154-2744f180-5a08-11eb-8c9c-3906ff8ccaa7.png)
Random forest with limited feature by using feature selection curve close to the perfect ROC curve have a better performance level than the others.
![eee](https://user-images.githubusercontent.com/47764275/105009171-2ca23c00-5a08-11eb-976c-859e69fc2144.png)
Kth nearest neighbor with limited feature by using feature extraction curve close to the perfect ROC curve have a better performance level than the others.

CONCLUSION

Using random forests has produced the best performances in test error rate and having true positives/true negatives


<img width="380" alt="gggg" src="https://user-images.githubusercontent.com/47764275/105009451-7d199980-5a08-11eb-9b9a-74a47b417b01.PNG">
<img width="381" alt="mmm" src="https://user-images.githubusercontent.com/47764275/105009476-83a81100-5a08-11eb-80f7-603d37225f70.PNG">
