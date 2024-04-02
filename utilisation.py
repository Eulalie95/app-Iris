var1= float(input('Mesure 1 : '))
var2= float(input('Mesure 2 : '))
var3= float(input('Mesure 3 : '))
var4= float(input('Mesure 4 : '))
import bentoml

Model= bentoml.sklearn.get("iris_model:latest").to_runner()
Model.init_local()
pred=Model.predict.run([[var1,var2,var3,var4]])
print(pred)
