from selenium import webdriver
from bs4 import BeautifulSoup
import time
import json
import random


def get_trail_urls(root_url, chrome_path, urls_txt=None):
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    options.add_argument("disable-infobars")
    options.add_argument("--disable-extensions")
    browser = webdriver.Chrome(executable_path=chrome_path, options=options)
    browser.get(root_url)
    input("Did you tell them you are not a robot?")
    if urls_txt:
        urls = txt_to_list(urls_txt)
    else:
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

def get_reviews_and_trail_metadata(url, browser):
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
    browser.get("https://www.alltrails.com" + url + "?u=i")
    # input("Did you tell them you are not a robot?")
    # load the comments....
    while True:
        try:
            time.sleep(random.randint(3, 10))
            browser.find_element_by_class_name("styles-module__button___1nuva").click()
        except:
            break
    soup = BeautifulSoup(browser.page_source)

    # get trail metadata
    meta = {}
    meta["description"] = soup.find("p", {"class":"xlate-google line-clamp-4", "id":"auto-overview"}).text
    meta["length_elev_type"] = [i.text for i in soup.findAll("span", {"class":"styles-module__detailData___kQ-eK"})]
    meta["tags"] = [i.text for i in soup.find("section", {"class":"tag-cloud"})]
    meta["coords"] = {i.get("itemprop"): i.get("content") for i in soup.find("div", {"itemprop":"geo"}).findAll("meta")}
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

    # combine both meta and reviews #
    scrape_outcome = {"reviews":reviews, "meta":meta}

    return scrape_outcome

def txt_to_list(fpath):
    with open(fpath, "r") as f:
        urls = f.readlines()
    return [url.split("\n")[0] for url in urls]

def list_to_txt(to_record, dest):
    with open(dest, "w") as f:
        for i in to_record:
            f.write(i+"\n")



if __name__ == "__main__":
    root_url = "https://www.alltrails.com/canada/nova-scotia?ref=search"
    urls_txt = r"C:\Users\NoahB\Desktop\School\first year MCSC (2021-2022)\CS6612\group_proj\GimmeAllTheTrails\data\trail_urls.txt"
    trail_urls, browser = get_trail_urls(root_url, urls_txt)
    time.sleep(random.randint(3, 6))
    total = {}
    problematic_urls = []
    for trail in trail_urls:
        time.sleep(random.randint(3, 6))
        try:
            total[trail] = get_reviews_and_trail_metadata(trail, browser)
        except:
            input("tell em ur not a robot")
            try:
                total[trail] = get_reviews_and_trail_metadata(trail, browser)
            except:
                print(f"error with {trail}")
                problematic_urls.append(trail)
                list_to_txt(problematic_urls, "bad_ones.txt")
            with open("aggregate3.json", "w") as f:
                json.dump(total, f)
    with open("aggregate3.json", "w") as f:
        json.dump(total, f)

