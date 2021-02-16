from typing import List


def count_ranking(lines: List):
    counter = {}
    for line in lines:
        tags = line.split()[1:]
        for tag in tags:
            if tag not in counter:
                counter[tag] = 0
            counter[tag] += 1
    rank_list = sorted(counter.items(), key=(lambda x: x[1]), reverse=True)
    rank_list = dict(rank_list[:100])
    print(rank_list)
    return rank_list


def english_tag_ranking():
    with open('E:\ADD\ADD\\NUS_WID_Tags\dataset\english_tags1.txt', 'r', encoding='utf-8') as english_tags_file:
        lines = english_tags_file.readlines()
        # print(len(lines))
        rank_list = count_ranking(lines)
        return rank_list


def final_tags_ranking():
    with open('E:\ADD\ADD\\NUS_WID_Tags\dataset\\final_tags.txt', 'r', encoding='utf-8') as final_tags_file:
        lines = final_tags_file.readlines()
        # print(len(lines))

        count_ranking(lines)
        # for i in range(500):
        #    print(c[i])


def flicker_tags_ranking():
    with open('E:\ADD\ADD\\NUS_WID_Tags\All_Tags.txt', 'r', encoding='utf-8') as final_tags_file:
        lines = final_tags_file.readlines()
        count_ranking(lines)


def tag_100_ranking():
    with open('E:\ADD\ADD\\NUS_WID_Tags\dataset\\rank_tags1.txt', 'r', encoding='utf-8') as final_tags_file:
        lines = final_tags_file.readlines()
        count_ranking(lines)
        print(len(lines))


def main():
    # english_tag_ranking()
    # final_tags_ranking()
    # flicker_tags_ranking()
    tag_100_ranking()


main()
