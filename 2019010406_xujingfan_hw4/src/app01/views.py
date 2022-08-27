from pickle import NONE
from tracemalloc import start
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from app01.train import lenet_train
from app01.lossplt import loss_plot
from app01.forms import ParaForm
from .models import task
import re
import datetime
import pytz
import threading

def hello(request):
    return render(request, 'main_page.html')

def long_process(epochs,batch_size,learning_rate,new_task,structure,optimizer):
    lenet_train(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, structure=structure, optimizer_name=optimizer, num_classes=10, task_name=str(new_task.id))
    loss_plot(epochs,str(new_task.id),batch_size)
    new_task.status = True
    new_task.save()


@csrf_exempt 
def train(request):
    if request.method == 'POST':
        form = ParaForm(data=request.POST)
        new_task=task()
        if form.is_valid():
            optimizer = form.cleaned_data['optimizer']
            structure = form.cleaned_data['structure']
            epochs = form.cleaned_data['epochs']
            batch_size = form.cleaned_data['batch_size']
            learning_rate = form.cleaned_data['learning_rate']
            new_task.task_name = form.cleaned_data['task_name']
            new_task.username = form.cleaned_data['username']
            new_task.status = False
            new_task.save()
            if epochs is None: epochs=10
            if batch_size is None: batch_size=235
            if learning_rate is None: learning_rate=0.001
        else:
            print(form.errors)
        t = threading.Thread(target=long_process,args=(epochs,batch_size,learning_rate,new_task,structure,optimizer))
        # t.setDaemon(True)
        t.start()
        return HttpResponseRedirect("/result/")
    else:
        return render(request, "train.html")

def result(request):
    task_list = task.objects.all()
    context = {'task_list':task_list}
    return render(request, 'result.html',context)

def detail(request):
    task_detail= str(request.path)
    task_id = re.findall("\d+",task_detail)
    task_id = int("".join(task_id))
    this_task = task.objects.get(id=task_id)
    task_id = str(task_id)
    
    if this_task.status == True:
        context = { "task_name":this_task.task_name,
                    "username": this_task.username,
                    "last_time":this_task.end_time-this_task.init_time, 
                    "status": this_task.status,
                    "logger_list":None,
                    "task_id":task_id}
        return render(request,"detail.html",context)
    elif this_task.status == False:
        
        starttime = this_task.init_time
        endtime = datetime.datetime.now(pytz.utc)
        logger=open('log/'+task_id+'.txt',mode='r',encoding="utf8")
        logger_list = logger.readlines()
        logger.close()

        context = { "task_name":this_task.task_name,
                    "username": this_task.username,
                    "last_time":endtime-starttime, 
                    "logger_list":logger_list,
                    "task_id":None}
        return render(request,"detail.html",context)