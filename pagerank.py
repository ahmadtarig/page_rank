import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    """
    sys.argv is a list of words written in the command line or terminal to run this file.
    len(sys.argv) represents the number of words.
    """
    if len(sys.argv) != 2:  # If this condition is true, it means the user did not select a directory.
        sys.exit("Usage: python pagerank.py corpus")  # Exit the file with this message.
    directory = sys.argv[1]  # Store the directory or name.
    corpus = crawl(directory)  # Parse a relation into a dictionary variable. Example:
    # {
    #  '1.html': {'2.html'},
    #  '2.html': {'3.html', '1.html'},
    #  '3.html': {'2.html', '4.html'},
    #  '4.html': {'2.html'}
    # }
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")  # :.4f to print only 4 places after the floating-point.
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
    # Step one: calculate the probability of linked pages.
    # Assume linked pages have equal probability to be chosen.
    # For each linked page, the probability = damping_factor / total number of linked pages.

    linked_pages = list(corpus[page])  # Store linked pages in a list.
    if len(linked_pages) > 0:  # If the list isn't empty.
        # Calculate probability for each page.
        transition_model_dict = {page: damping_factor / len(linked_pages) for page in linked_pages}
    else:
        # Initialize an empty dictionary.
        transition_model_dict = {}
        damping_factor = 0

    # Step two: calculate the probability of other pages.
    # Assume other pages have equal probability to be chosen.
    # For each non-linked page, the probability = (1 - damping_factor) / total number of non-linked pages.
    # By using these two equations, the sum of probabilities equals 1.

    # This line stores all non-linked pages in a list.
    non_linked_pages = [non_linked_page for non_linked_page in corpus.keys() if non_linked_page not in linked_pages]
    if len(non_linked_pages) > 0:
        transition_model_dict.update(
            {page: (1 - damping_factor) / len(non_linked_pages) for page in non_linked_pages})
    else:
        """
        Return to calculating linked pages with a new damping_factor = 1
        Using recursion in this case is the best option.
        """
        if damping_factor != 0:
            transition_model(corpus, page, 1)

    return transition_model_dict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to the transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    # Step one:

    # Initialize PageRank values for each page as 0.
    pagerank = {page: 0 for page in corpus}

    # Simulate `n` random page visits according to the transition model.
    # `damping_factor`: probability of choosing a linked page.
    for _ in range(n):

        # Choose a random page to start.
        page = random.choice(list(corpus.keys()))

        while True:
            # Add a visit to the current page.
            pagerank[page] += 1

            # Explore other linked pages if a random number is less than the damping factor.
            # Otherwise, break the loop.
            if random.random() < damping_factor:
                linked_pages = corpus[page]

                if linked_pages:  # Check if the list isn't empty.
                    page = random.choice(list(linked_pages))  # Choose a random linked page.
                else:
                    page = random.choice(list(corpus.keys()))  # Choose a random page from all pages.
            else:
                break

    # Step two: Normalize page scores.
    total_score = sum(list(pagerank.values()))  # Calculate the sum of page scores.
    pages_rank = {page: score / total_score for page, score in pagerank.items()}
    # Create a PageRank dictionary where values are normalized scores.
    return pages_rank  # Return the PageRank dictionary.


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Step one:
    # Calculate ranks using the transition_model.
    pagerank = {page: 0 for page in corpus.keys()}  # Initialize a dictionary with keys as pages and values as 0.
    for page in corpus.keys():  # Iterate through all pages in the corpus.
        transition_model_dict = transition_model(corpus, page, damping_factor)  # Use the transition_model function.
        for key, value in transition_model_dict.items():  # Iterate through the transition model's output.
            pagerank[key] += value  # Increment the rank for each page in the linked pages.

    # Step two:
    # Calculate ranks using previous knowledge, pagerank.
    # Initialize a dictionary with keys as pages and values as (1 - damping_factor) / len(corpus),
    # representing the probability of reaching each page.
    new_pagerank = {page: (1 - damping_factor) / len(corpus) for page in corpus.keys()}
    for page, linked_pages in corpus.items():  # Iterate through corpus items.
        for linked_page in linked_pages:  # Iterate through linked pages.
            new_pagerank[linked_page] += pagerank[page] * damping_factor / len(linked_pages)
            # Update the PageRank using the equation: pagerank(page) * damping_factor / number of linked pages.

    # Step three:
    # Normalize the pagerank values to be between 0 and 1.
    total = sum(new_pagerank.values())  # Calculate the total score.
    new_pagerank = {page: rank / total for page, rank in new_pagerank.items()}  # Divide each score by the total score.

    return new_pagerank


if __name__ == "__main__":
    main()
