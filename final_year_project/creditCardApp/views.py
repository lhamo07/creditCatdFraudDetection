import numpy as np
from django.contrib import auth, messages
from django.contrib.auth.forms import AuthenticationForm
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from .form import CreateUserForm
from django.contrib.auth import authenticate, login,logout
from django.shortcuts import render

def home(request):
    return render(request,"header.html")

def loginPage(request):
    if request.method == "POST":
       username=request.POST.get('username')
       password=request.POST.get('password')

       user=authenticate(request,username=username,password=password)
       if user is not None:
           login(request,user)
           return redirect('index')
       else:
           messages.info(request,'Username or password is incorrect')
    context={}
    return render(request, 'login.html', context)

def about(request):
    return render(request,'about.html')
def index(request):
    return render(request,'index.html')

def uploadFile(request):
    context={}
    if request.method=='POST':
        uploaded_file=request.FILES['document']
        fs=FileSystemStorage();
        name=fs.save(uploaded_file.name,uploaded_file)
        context['url']=fs.url(name)



    return render(request,'upload.html',context)
def showReport(request):
    return render(request,'report.html')



def logoutUser(request):
    logout(request)
    messages.info(request, "You have successfully logged out.")

    return redirect('login')

def register(request):
    form = CreateUserForm()
    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            form.save()
            user=form.cleaned_data.get('username')
            messages.success(request,'Account was created for '+user)
            return redirect('login')
    else:
        form=CreateUserForm()
    context = {'form': form}
    return render(request, 'register.html', context)
def list(request):
    return render(request,'list.html')
# def output(request):
#
#     return render(request,'dataset.html')
# def homeData(request):
#     return render(request,'homeData.html')
def getPrediction(time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount
               ):
    import pickle
    model=pickle.load(open("credit_card.pkl","rb"))

    prediction=model.predict(np.array([[time,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,amount]]))
    if prediction ==0:
        return "Normal Transaction"
    elif prediction==1:

        return "Fraud Transaction"
    else:
        return "error"
    # return render(request,'indexx.html')
def result(request):
    time=float(request.GET['n1'])
    v1 = float(request.GET['n2'])
    v2 = float(request.GET['n3'])
    v3 = float(request.GET['n4'])
    v4 = float(request.GET['n5'])
    v5 = float(request.GET['n6'])
    v6 = float(request.GET['n7'])
    v7= float(request.GET['n8'])
    v8 = float(request.GET['n9'])
    v9 = float(request.GET['n10'])
    v10= float(request.GET['n11'])
    v11= float(request.GET['n12'])
    v12= float(request.GET['n13'])
    v13= float(request.GET['n14'])
    v14= float(request.GET['n15'])
    v15= float(request.GET['n16'])
    v16= float(request.GET['n17'])
    v17= float(request.GET['n18'])
    v18= float(request.GET['n19'])
    v19= float(request.GET['n20'])
    v20= float(request.GET['n21'])
    v21= float(request.GET['n22'])
    v22= float(request.GET['n23'])
    v23= float(request.GET['n24'])
    v24= float(request.GET['n25'])
    v25= float(request.GET['n26'])
    v26= float(request.GET['n27'])
    v27= float(request.GET['n28'])
    v28= float(request.GET['n29'])
    amount = float(request.GET['n30'])

    result=getPrediction(time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount)

    # pred=model.predict(np.array([val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,
    #                      val11,val12,val13,val14,val15,val16,val17,val18,val19,val20,
    #                      val21,val22,val23,val24,val25,val26,val27,val28,val29,val30]).reshape(1,-1))
    # result1 = " "
    # if pred == [0]:
    #     result1: "fraud"
    # else:
    #     result1: "legit"

    return render(request ,'index.html',{"result":result})






