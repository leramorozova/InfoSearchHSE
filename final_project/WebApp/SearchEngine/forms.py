from django import forms


class SearchForm(forms.Form):
    fields = forms.CharField(label='lemma', max_length=200)
