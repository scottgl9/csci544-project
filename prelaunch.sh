mkdir -p source/data/wordlist
mkdir -p source/out

curl https://raw.githubusercontent.com/uclanlp/gn_glove/master/wordlist/male_word_file.txt -o source/data/wordlist/male_word_file.txt
curl https://raw.githubusercontent.com/uclanlp/gn_glove/master/wordlist/female_word_file.txt -o source/data/wordlist/female_word_file.txt