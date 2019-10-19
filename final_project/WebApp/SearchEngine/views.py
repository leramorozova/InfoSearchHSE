from django.shortcuts import render
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from .forms import SearchForm
from .SearchUtilities.search_model_base import WordToVecSearch, ElmoSearch, TfIdfSearch, Bm25Search


def index(request):
    form = SearchForm()
    return render(request, 'SearchEngine/index.html', {'form': form})


def search_result(request):
    lemma = request.GET['lemma']
    search_type = request.GET['tabs']
    if search_type == "tfidf":
        model = TfIdfSearch()
    elif search_type == "cit":
        model = Bm25Search()
    elif search_type == "source":
        model = ElmoSearch()
    else:
        model = WordToVecSearch()
    search_res = model.search(lemma)
    page = request.GET.get('page')
    if search_res is None:
        return render(request, 'SearchEngine/failed_result.html', {"lemma": lemma})
    paginator = Paginator(search_res, 5)
    try:
        response = paginator.page(page)
    except PageNotAnInteger:
        response = paginator.page(1)
    except EmptyPage:
        response = paginator.page(paginator.num_pages)
    return render(request, 'SearchEngine/search_result.html', {"lemma": lemma, "response": response,
                                                               "search_type": search_type})
