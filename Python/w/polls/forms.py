from django import forms

class PairGameForm(forms.Form):
  enter_a_code = forms.CharField(max_length=100)
