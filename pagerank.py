import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    """
    sys.argv is list of words wrote in cmd or terminal to run this file
    len(sys.argv) mean number of words
    """
    if len(sys.argv) != 2:  # if condition true that means user does not select direction
        sys.exit("Usage: python pagerank.py corpus")  # exit file with this massage
    direction = sys.argv[1]  # store direction or  name
    corpus = crawl(direction)  # parce a relation in dict variable example :
    #  {
    #  '1.html': {'2.html'},
    #  '2.html': {'3.html', '1.html'},
    #  '3.html': {'2.html', '4.html'},
    #  '4.html': {'2.html'}
    #  }
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")  # :.4f to print only 4 place after floating-point
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    # step one : calc probability of liked pages
    # assume linked page have equal probability to choice
    # for each linked page probability of page = damping_factor/total number of linked page

    linked_page = list(corpus[page])  # store linked pages in list
    if len(linked_page) > 0:  # if list isn't null
        # calc probability for each pages
        transition_model_dict = {page: damping_factor / len(linked_page) for page in linked_page}
    else:
        # initialize an empty dict
        transition_model_dict = {}
        damping_factor = 0

    # step two : calc probability of other pages
    # same thing assume other page have equal probability to choice
    # for each non_linked page probability of page = (1 - damping_factor)/ total number of none linked page
    # by using this two equation sum of probability = 1

    # this condition line store all none_linked_page in list
    none_linked_pages = [none_linked_page for none_linked_page in corpus.keys() if none_linked_page not in linked_page]
    if len(none_linked_pages) > 0:
        transition_model_dict.update(
            {page: (1 - damping_factor) / len(none_linked_pages) for page in none_linked_pages})
    else:
        """
            back to calc a linked pages with new damping_factor = 1
            Using recursion in this case is the best option
        """
        if damping_factor != 0:
            transition_model(corpus, page, 1)

    return transition_model_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # step one:

    # page rank is dict key: represent page name and value: represent score
    # as initial step score for all pages = 0
    pagerank = {page: 0 for page in corpus}

    # simulate a 5000 or n sample of search process
    # damping_factor : mean probability of none random page selection
    for _ in range(n):

        # choice random page name
        page = random.choice(list(corpus.keys()))

        while True:
            # At first: add visit to current page
            pagerank[page] += 1

            # In normal state a user visit other pages that are linked to from the current page
            # a while loop simulate this behavior :

            # if a random number is less than dumping factor
            # 1- select another page that current page linked to
            # 2- continue loop with the new page

            # else break loop

            if random.random() < damping_factor:
                # linked_pages store a list of linked page
                # Value of page key in corpus dict
                linked_pages = corpus[page]

                if linked_pages:  # check if list isn't null
                    # choice random page from list
                    page = random.choice(list(linked_pages))
                else:
                    # else select random page from corpus
                    page = random.choice(list(corpus.keys()))
            else:
                break

    # step two: normalize pages score
    total_score = sum(list(pagerank.values()))  # calc sum of pages score
    pages_rank = {page: score/total_score for page, score in pagerank.items()}
    # create a page rank dict
    # keys : is list of pages
    # value : is previous score divide by total score
    return pages_rank  # return dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # step one: calc a base probability of each page

    # 1 - damping_factor / total number of pages
    # base rank based on one rule all page have same probability * 1 - damping_factor
    base_rank_score = (1 - damping_factor) / len(corpus)
    base_rank = {page: base_rank_score for page in corpus.keys()}
    # link rank based on probability to reach page from other page
    link_rank = {page: 0 for page in corpus.keys()}  # initial score is 0 for all pages

    # step two: clac link rank

    # each page have probability to reach page from p(page1) + p(page2) , ...
    # p(page) = damping_factor * probability of reach current page / number of linked page
    # loop for all corpus items
    for current_page, linked_pages in corpus.items():
        # nested loop
        for page in linked_pages:
            # increment a link rank for each linked_page using previous probability equation
            link_rank[page] += base_rank[current_page] * damping_factor / len(linked_pages)

    # step three: calc total_rank
    # total_rank for each page = base_rank_score + link_rank_score(page)
    total_rank = {page: link_rank_score + base_rank_score for page, link_rank_score in link_rank.items()}

    # step four: normalize a score (from 0 to 1)
    total_score = sum(total_rank.values())  # calc total score
    total_rank = {page: score / total_score for page, score in total_rank.items()}  # score /= total score

    return total_rank


if __name__ == "__main__":
    main()
