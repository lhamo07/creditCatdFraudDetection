# from django.db import models
# from sklearn.linear_model import LogisticRegression
# import joblib
# # # Create your models here.
# #
# # # class LoginForm(models.Model):
# # #     Username='admin'
# # #     Password='admin'
# # #     username=models.CharField(max_length=20,blank=False,null=False)
# # #     password=models.TextField(max_length=20,blank=False,null=False)
# Class=(
#     (0,'legit'),
#     (1,'fraud'),
# )
# class Data(models.Model):
#     V1=models.FloatField()
#     V2 = models.FloatField()
#     V3 = models.FloatField()
#     V4 = models.FloatField()
#     V5 = models.FloatField()
#     V6 = models.FloatField()
#     V7 = models.FloatField()
#     V8 = models.FloatField()
#     V9 = models.FloatField()
#     V10 = models.FloatField()
#     V11 = models.FloatField()
#     V12 = models.FloatField()
#     V13 = models.FloatField()
#     V14 = models.FloatField()
#     V15= models.FloatField()
#     V16 =models.FloatField()
#     V17 = models.FloatField()
#     V18 = models.FloatField()
#     V19 = models.FloatField()
#     V20 = models.FloatField()
#     V21= models.FloatField()
#     V22= models.FloatField()
#     V23= models.FloatField()
#     V24= models.FloatField()
#     V25 = models.FloatField()
#     V26= models.FloatField()
#     V27= models.FloatField()
#     V28= models.FloatField()
#     Amount= models.FloatField()
#     predictions=models.IntegerField(blank=True)
#
#     def save(self,*args,**kwargs):
#         ml_model=joblib.load('finalized_model.joblib')
#         self.predictions=ml_model.predict([[self.V1,self.V2,self.V3,self.V4,self.V5,self.V6,self.V7,self.V8,self.V9,self.V10,
#                                             self.V11,self.V12,self.V13,self.V14,self.V15,self.V16,self.V17,self.V18,self.V19,self.V20,self.V21,
#                                             self.V22,self.V23,self.V24,self.V25,self.V26,self.V27,self.V28,self.Amount]])
#         return super().save(*args,**kwargs)
#
#
#
#
#
#
#
#
