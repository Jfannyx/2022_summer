from django import forms

class ParaForm(forms.Form):
    structure = forms.CharField(label='structure')
    optimizer = forms.CharField(label='optimizer')
    epochs = forms.IntegerField(label='epochs',initial=10,required=False)
    batch_size = forms.IntegerField(label='batch_size',initial=235,required=False)
    learning_rate = forms.FloatField(label='learning_rate',initial=0.01,required=False)
    task_name = forms.CharField(label='task_name')
    username = forms.CharField(label='username')