from selenium import webdriver
from bs4 import BeautifulSoup
import time
import json
import random


def get_trail_urls(root_url, chrome_path):
    """

    :param root_url: url of area of interest
    :param chrome_path: path to local chromedriver.exe
    :return: return list of urls and the browser
    """
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")
    browser = webdriver.Chrome(executable_path=chrome_path, options=options)
    browser.get(root_url)
    input("Did you tell them you are not a robot?")
    while True:
        try:
            browser.find_element_by_class_name("styles-module__button___1nuva ").click()
            time.sleep(random.randint(2, 7))
        except:
            break
    soup = BeautifulSoup(browser.page_source)
    cards = soup.find("div", {"class":"styles-module__results___24LBd"}).contents
    urls = [card.get("itemid") for card in cards]
    return urls, browser

def get_reviews(url, browser):
    """
    We should be able to get each individual hike using the url
    e.g.
    https://www.alltrails.com/trail/canada/nova-scotia/cape-split-trail

    reviews are first in <div class="styles-module__tabContainer___2wEWm">
     within this for each user, the tags for the hike are in
     class="styles-module__info___1Mbn6> styles-module__infoTrailDetails___23Xx3"
     where each span in that section can be identified by its title

     the full review cna be found in:

     class="styles-module__details___1QPxR xlate-google"
     where
     <p itemprop="reviewBody"> contains the information

    :return:
    """
    browser.get("https://www.alltrails.com" + url)
    while True:
        try:
            time.sleep(random.randint(3, 10))
            browser.find_element_by_class_name("styles-module__button___1nuva").click()

        except:
            break
    soup = BeautifulSoup(browser.page_source)
    # contents-> reviews -> feed-items null
    reviews = {}
    all_reviews = soup.find("div", {"class": "styles-module__tabContainer___2wEWm"}).contents[0].contents[1]
    reviews["written"] = [r.text for r in all_reviews.findAll("p", {"itemprop":"reviewBody"})]

    # get key words
    key_words = all_reviews.findAll("div", {"class": "styles-module__info___1Mbn6> styles-module__infoTrailDetails___23Xx3"})
    key_words = [k.findAll("span", {"class": "styles-module__tag___2s-oD styles-module__activityTag___3-RdN"}) for k in key_words]
    cleaned_key_words = []
    for kws in key_words:
        words = []
        for word in kws:
            words.append(word.get("title"))
        cleaned_key_words.append(words)

    reviews['key_words'] = cleaned_key_words
    # get ratings from the user
    ratings = [i.get("aria-label") for i in all_reviews.findAll("span", {"class":"MuiRating-root default-module__rating___1k45X MuiRating-sizeLarge MuiRating-readOnly"})]
    reviews["ratings"] = ratings
    return reviews

if __name__ == "__main__":
    """
    running test
    """
    root_url = "https://www.alltrails.com/canada/nova-scotia?ref=search"
    trail_urls, browser = get_trail_urls(root_url)
    time.sleep(random.randint(3, 10))
    url = "https://www.alltrails.com/trail/canada/nova-scotia/cape-split-trail"
    total = {}
    for trail in trail_urls:
        time.sleep(random.randint(3, 10))
        try:
            total[trail] = get_reviews(trail, browser)
        except:
            input("tell em ur not a robot")
            total[trail] = get_reviews(trail, browser)
            with open("aggregate.json", "w") as f:
                json.dump(total, f)
    with open("aggregate.json", "w") as f:
        json.dump(total, f)

