# file for preprocessing


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def read_dem_files():
    dem_files ="""cookie_dem/001-0.cha  cookie_dem/070-2.cha  cookie_dem/205-1.cha  cookie_dem/310-0.cha  cookie_dem/511-0.cha
    cookie_dem/001-2.cha  cookie_dem/076-0.cha  cookie_dem/206-0.cha  cookie_dem/311-0.cha  cookie_dem/515-1.cha
    cookie_dem/003-0.cha  cookie_dem/076-2.cha  cookie_dem/207-0.cha  cookie_dem/319-0.cha  cookie_dem/526-1.cha
    cookie_dem/005-0.cha  cookie_dem/076-4.cha  cookie_dem/212-0.cha  cookie_dem/325-0.cha  cookie_dem/527-0.cha
    cookie_dem/005-2.cha  cookie_dem/078-0.cha  cookie_dem/212-1.cha  cookie_dem/325-1.cha  cookie_dem/527-1.cha
    cookie_dem/007-1.cha  cookie_dem/078-1.cha  cookie_dem/212-2.cha  cookie_dem/329-0.cha  cookie_dem/528-0.cha
    cookie_dem/007-3.cha  cookie_dem/087-1.cha  cookie_dem/212-3.cha  cookie_dem/334-0.cha  cookie_dem/529-0.cha
    cookie_dem/010-0.cha  cookie_dem/089-0.cha  cookie_dem/213-1.cha  cookie_dem/334-1.cha  cookie_dem/530-0.cha
    cookie_dem/010-1.cha  cookie_dem/091-0.cha  cookie_dem/213-2.cha  cookie_dem/337-0.cha  cookie_dem/539-0.cha
    cookie_dem/010-2.cha  cookie_dem/091-1.cha  cookie_dem/213-3.cha  cookie_dem/338-0.cha  cookie_dem/544-0.cha
    cookie_dem/010-3.cha  cookie_dem/091-2.cha  cookie_dem/216-0.cha  cookie_dem/339-0.cha  cookie_dem/544-1.cha
    cookie_dem/010-4.cha  cookie_dem/094-1.cha  cookie_dem/216-1.cha  cookie_dem/341-0.cha  cookie_dem/551-0.cha
    cookie_dem/014-2.cha  cookie_dem/094-2.cha  cookie_dem/218-0.cha  cookie_dem/342-0.cha  cookie_dem/559-0.cha
    cookie_dem/016-0.cha  cookie_dem/094-3.cha  cookie_dem/218-1.cha  cookie_dem/342-1.cha  cookie_dem/562-0.cha
    cookie_dem/016-1.cha  cookie_dem/097-1.cha  cookie_dem/220-0.cha  cookie_dem/343-0.cha  cookie_dem/563-0.cha
    cookie_dem/016-3.cha  cookie_dem/120-0.cha  cookie_dem/220-1.cha  cookie_dem/344-0.cha  cookie_dem/573-0.cha
    cookie_dem/016-4.cha  cookie_dem/120-1.cha  cookie_dem/221-0.cha  cookie_dem/344-2.cha  cookie_dem/578-0.cha
    cookie_dem/018-0.cha  cookie_dem/120-2.cha  cookie_dem/221-1.cha  cookie_dem/346-0.cha  cookie_dem/579-0.cha
    cookie_dem/023-0.cha  cookie_dem/120-3.cha  cookie_dem/221-2.cha  cookie_dem/349-0.cha  cookie_dem/580-0.cha
    cookie_dem/023-2.cha  cookie_dem/120-4.cha  cookie_dem/221-3.cha  cookie_dem/349-1.cha  cookie_dem/581-0.cha
    cookie_dem/024-1.cha  cookie_dem/122-0.cha  cookie_dem/222-0.cha  cookie_dem/350-0.cha  cookie_dem/585-0.cha
    cookie_dem/024-2.cha  cookie_dem/122-1.cha  cookie_dem/222-1.cha  cookie_dem/350-1.cha  cookie_dem/587-0.cha
    cookie_dem/029-0.cha  cookie_dem/125-0.cha  cookie_dem/223-0.cha  cookie_dem/354-0.cha  cookie_dem/591-0.cha
    cookie_dem/029-1.cha  cookie_dem/127-0.cha  cookie_dem/223-1.cha  cookie_dem/355-0.cha  cookie_dem/592-0.cha
    cookie_dem/030-0.cha  cookie_dem/134-0.cha  cookie_dem/226-0.cha  cookie_dem/355-1.cha  cookie_dem/594-0.cha
    cookie_dem/030-1.cha  cookie_dem/134-1.cha  cookie_dem/234-0.cha  cookie_dem/356-0.cha  cookie_dem/595-0.cha
    cookie_dem/033-0.cha  cookie_dem/134-2.cha  cookie_dem/235-0.cha  cookie_dem/356-1.cha  cookie_dem/598-0.cha
    cookie_dem/033-1.cha  cookie_dem/134-3.cha  cookie_dem/235-2.cha  cookie_dem/357-0.cha  cookie_dem/601-0.cha
    cookie_dem/033-2.cha  cookie_dem/144-0.cha  cookie_dem/236-0.cha  cookie_dem/358-0.cha  cookie_dem/607-0.cha
    cookie_dem/033-3.cha  cookie_dem/144-1.cha  cookie_dem/237-1.cha  cookie_dem/358-1.cha  cookie_dem/609-0.cha
    cookie_dem/033-4.cha  cookie_dem/148-0.cha  cookie_dem/237-2.cha  cookie_dem/360-0.cha  cookie_dem/610-0.cha
    cookie_dem/035-0.cha  cookie_dem/154-0.cha  cookie_dem/238-0.cha  cookie_dem/361-0.cha  cookie_dem/615-0.cha
    cookie_dem/035-1.cha  cookie_dem/154-1.cha  cookie_dem/244-0.cha  cookie_dem/362-0.cha  cookie_dem/620-0.cha
    cookie_dem/039-0.cha  cookie_dem/157-0.cha  cookie_dem/247-0.cha  cookie_dem/362-1.cha  cookie_dem/624-0.cha
    cookie_dem/043-0.cha  cookie_dem/157-1.cha  cookie_dem/252-0.cha  cookie_dem/368-0.cha  cookie_dem/635-0.cha
    cookie_dem/046-0.cha  cookie_dem/157-2.cha  cookie_dem/252-1.cha  cookie_dem/369-0.cha  cookie_dem/636-0.cha
    cookie_dem/046-2.cha  cookie_dem/164-1.cha  cookie_dem/252-2.cha  cookie_dem/381-0.cha  cookie_dem/639-0.cha
    cookie_dem/049-0.cha  cookie_dem/164-2.cha  cookie_dem/257-0.cha  cookie_dem/381-1.cha  cookie_dem/640-0.cha
    cookie_dem/049-1.cha  cookie_dem/164-3.cha  cookie_dem/257-2.cha  cookie_dem/450-0.cha  cookie_dem/642-0.cha
    cookie_dem/050-0.cha  cookie_dem/168-0.cha  cookie_dem/260-1.cha  cookie_dem/450-1.cha  cookie_dem/648-0.cha
    cookie_dem/051-0.cha  cookie_dem/168-1.cha  cookie_dem/260-2.cha  cookie_dem/458-0.cha  cookie_dem/650-0.cha
    cookie_dem/051-1.cha  cookie_dem/172-1.cha  cookie_dem/264-0.cha  cookie_dem/461-0.cha  cookie_dem/651-0.cha
    cookie_dem/051-2.cha  cookie_dem/172-2.cha  cookie_dem/268-0.cha  cookie_dem/465-0.cha  cookie_dem/656-0.cha
    cookie_dem/051-3.cha  cookie_dem/172-3.cha  cookie_dem/269-0.cha  cookie_dem/466-0.cha  cookie_dem/657-0.cha
    cookie_dem/053-1.cha  cookie_dem/173-1.cha  cookie_dem/269-1.cha  cookie_dem/466-1.cha  cookie_dem/660-0.cha
    cookie_dem/057-0.cha  cookie_dem/178-0.cha  cookie_dem/270-0.cha  cookie_dem/468-0.cha  cookie_dem/663-0.cha
    cookie_dem/057-1.cha  cookie_dem/178-1.cha  cookie_dem/270-1.cha  cookie_dem/470-1.cha  cookie_dem/672-0.cha
    cookie_dem/057-2.cha  cookie_dem/181-0.cha  cookie_dem/270-2.cha  cookie_dem/471-0.cha  cookie_dem/674-0.cha
    cookie_dem/058-0.cha  cookie_dem/181-1.cha  cookie_dem/271-2.cha  cookie_dem/472-0.cha  cookie_dem/676-0.cha
    cookie_dem/058-1.cha  cookie_dem/181-2.cha  cookie_dem/276-0.cha  cookie_dem/474-0.cha  cookie_dem/681-0.cha
    cookie_dem/058-3.cha  cookie_dem/181-3.cha  cookie_dem/279-0.cha  cookie_dem/476-0.cha  cookie_dem/689-0.cha
    cookie_dem/058-4.cha  cookie_dem/183-0.cha  cookie_dem/279-1.cha  cookie_dem/488-0.cha  cookie_dem/690-0.cha
    cookie_dem/061-0.cha  cookie_dem/183-1.cha  cookie_dem/282-0.cha  cookie_dem/488-1.cha  cookie_dem/695-0.cha
    cookie_dem/061-1.cha  cookie_dem/183-2.cha  cookie_dem/282-1.cha  cookie_dem/492-0.cha  cookie_dem/698-0.cha
    cookie_dem/062-0.cha  cookie_dem/183-3.cha  cookie_dem/282-2.cha  cookie_dem/493-0.cha  cookie_dem/702-0.cha
    cookie_dem/062-3.cha  cookie_dem/184-0.cha  cookie_dem/283-0.cha  cookie_dem/493-1.cha  cookie_dem/703-0.cha
    cookie_dem/065-0.cha  cookie_dem/184-1.cha  cookie_dem/283-1.cha  cookie_dem/497-0.cha  cookie_dem/704-0.cha
    cookie_dem/065-1.cha  cookie_dem/184-2.cha  cookie_dem/289-2.cha  cookie_dem/497-1.cha  cookie_dem/705-0.cha
    cookie_dem/065-2.cha  cookie_dem/190-1.cha  cookie_dem/291-1.cha  cookie_dem/504-0.cha  cookie_dem/707-0.cha
    cookie_dem/066-0.cha  cookie_dem/190-2.cha  cookie_dem/291-2.cha  cookie_dem/506-0.cha  cookie_dem/711-0.cha
    cookie_dem/067-1.cha  cookie_dem/203-0.cha  cookie_dem/293-1.cha  cookie_dem/508-0.cha  cookie_dem/714-0.cha
    cookie_dem/067-2.cha  cookie_dem/203-1.cha  cookie_dem/306-0.cha  cookie_dem/508-1.cha"""
    files = dem_files.split()
    dems = []
    for file in files:
        cha_file = open(file, 'r')
        lines = cha_file.readlines()
        
        pat_lines = []
        for i in range(len(lines)):
            if lines[i][0:5] == "*PAR:" or (lines[i][0:1] == "\t" and lines[i-1][0:5] == "*PAR:"):
                line = lines[i].split()
                if line[0] == "*PAR:":
                    
                    if len(line[-1]) > 10 and (line[-1][2] in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]):
                        new_line = line[1:-1]
                    else:
                        new_line = line[1:]
                else:
                    new_line = line[:-1]
                end_line = []
                for word in new_line:
                    if word !="[+":
                        end_line += split_word(word)

                pat_lines +=end_line

        dems.append(pat_lines)
        cha_file.close()
    return dems

def read_con_files():
    con_files ="""cookie_control/002-0.cha  cookie_control/086-1.cha  cookie_control/141-0.cha  cookie_control/243-1.cha
cookie_control/002-1.cha  cookie_control/086-2.cha  cookie_control/141-1.cha  cookie_control/245-0.cha
cookie_control/002-2.cha  cookie_control/086-3.cha  cookie_control/141-2.cha  cookie_control/245-1.cha
cookie_control/002-3.cha  cookie_control/086-4.cha  cookie_control/141-3.cha  cookie_control/245-2.cha
cookie_control/006-2.cha  cookie_control/092-0.cha  cookie_control/142-0.cha  cookie_control/248-0.cha
cookie_control/006-3.cha  cookie_control/092-1.cha  cookie_control/142-1.cha  cookie_control/248-1.cha
cookie_control/006-4.cha  cookie_control/092-2.cha  cookie_control/142-3.cha  cookie_control/248-2.cha
cookie_control/013-0.cha  cookie_control/092-3.cha  cookie_control/143-3.cha  cookie_control/255-0.cha
cookie_control/013-2.cha  cookie_control/093-0.cha  cookie_control/145-1.cha  cookie_control/255-1.cha
cookie_control/013-3.cha  cookie_control/093-1.cha  cookie_control/145-3.cha  cookie_control/256-0.cha
cookie_control/013-4.cha  cookie_control/096-1.cha  cookie_control/146-1.cha  cookie_control/256-1.cha
cookie_control/015-0.cha  cookie_control/096-2.cha  cookie_control/150-0.cha  cookie_control/256-2.cha
cookie_control/015-1.cha  cookie_control/105-0.cha  cookie_control/150-1.cha  cookie_control/266-0.cha
cookie_control/015-2.cha  cookie_control/105-1.cha  cookie_control/150-2.cha  cookie_control/266-1.cha
cookie_control/015-3.cha  cookie_control/105-2.cha  cookie_control/155-0.cha  cookie_control/266-2.cha
cookie_control/015-4.cha  cookie_control/107-1.cha  cookie_control/155-2.cha  cookie_control/267-0.cha
cookie_control/017-4.cha  cookie_control/107-2.cha  cookie_control/155-3.cha  cookie_control/267-2.cha
cookie_control/021-0.cha  cookie_control/109-1.cha  cookie_control/158-0.cha  cookie_control/274-0.cha
cookie_control/021-1.cha  cookie_control/109-3.cha  cookie_control/158-1.cha  cookie_control/274-1.cha
cookie_control/021-2.cha  cookie_control/109-4.cha  cookie_control/158-2.cha  cookie_control/274-2.cha
cookie_control/021-3.cha  cookie_control/113-0.cha  cookie_control/158-3.cha  cookie_control/275-0.cha
cookie_control/021-4.cha  cookie_control/113-1.cha  cookie_control/166-0.cha  cookie_control/275-1.cha
cookie_control/022-0.cha  cookie_control/113-2.cha  cookie_control/166-1.cha  cookie_control/280-0.cha
cookie_control/022-1.cha  cookie_control/113-3.cha  cookie_control/166-2.cha  cookie_control/280-1.cha
cookie_control/022-2.cha  cookie_control/114-0.cha  cookie_control/167-1.cha  cookie_control/280-2.cha
cookie_control/028-1.cha  cookie_control/114-1.cha  cookie_control/167-2.cha  cookie_control/292-1.cha
cookie_control/028-4.cha  cookie_control/114-2.cha  cookie_control/167-3.cha  cookie_control/295-0.cha
cookie_control/034-0.cha  cookie_control/114-3.cha  cookie_control/171-0.cha  cookie_control/295-1.cha
cookie_control/034-1.cha  cookie_control/114-4.cha  cookie_control/171-1.cha  cookie_control/296-0.cha
cookie_control/034-2.cha  cookie_control/118-0.cha  cookie_control/172-0.cha  cookie_control/296-1.cha
cookie_control/034-3.cha  cookie_control/118-1.cha  cookie_control/175-0.cha  cookie_control/296-2.cha
cookie_control/034-4.cha  cookie_control/118-2.cha  cookie_control/175-1.cha  cookie_control/297-1.cha
cookie_control/042-1.cha  cookie_control/118-3.cha  cookie_control/175-2.cha  cookie_control/297-2.cha
cookie_control/042-2.cha  cookie_control/118-4.cha  cookie_control/175-3.cha  cookie_control/298-1.cha
cookie_control/042-3.cha  cookie_control/121-0.cha  cookie_control/182-3.cha  cookie_control/299-1.cha
cookie_control/042-4.cha  cookie_control/121-1.cha  cookie_control/192-0.cha  cookie_control/302-0.cha
cookie_control/045-0.cha  cookie_control/121-2.cha  cookie_control/192-2.cha  cookie_control/304-1.cha
cookie_control/045-2.cha  cookie_control/121-3.cha  cookie_control/196-0.cha  cookie_control/304-2.cha
cookie_control/045-3.cha  cookie_control/121-4.cha  cookie_control/196-1.cha  cookie_control/318-0.cha
cookie_control/052-0.cha  cookie_control/124-0.cha  cookie_control/208-0.cha  cookie_control/318-1.cha
cookie_control/052-2.cha  cookie_control/124-1.cha  cookie_control/208-1.cha  cookie_control/318-2.cha
cookie_control/054-0.cha  cookie_control/128-1.cha  cookie_control/208-2.cha  cookie_control/322-1.cha
cookie_control/055-0.cha  cookie_control/128-2.cha  cookie_control/209-1.cha  cookie_control/322-2.cha
cookie_control/056-0.cha  cookie_control/128-3.cha  cookie_control/209-2.cha  cookie_control/323-0.cha
cookie_control/056-3.cha  cookie_control/129-1.cha  cookie_control/209-3.cha  cookie_control/323-1.cha
cookie_control/056-4.cha  cookie_control/130-1.cha  cookie_control/210-1.cha  cookie_control/332-0.cha
cookie_control/059-2.cha  cookie_control/130-2.cha  cookie_control/210-2.cha  cookie_control/336-1.cha
cookie_control/059-3.cha  cookie_control/130-3.cha  cookie_control/211-1.cha  cookie_control/340-0.cha
cookie_control/059-4.cha  cookie_control/132-0.cha  cookie_control/211-2.cha  cookie_control/612-0.cha
cookie_control/068-0.cha  cookie_control/132-1.cha  cookie_control/225-0.cha  cookie_control/627-0.cha
cookie_control/068-2.cha  cookie_control/137-0.cha  cookie_control/225-2.cha  cookie_control/631-0.cha
cookie_control/068-3.cha  cookie_control/137-1.cha  cookie_control/227-0.cha  cookie_control/661-0.cha
cookie_control/071-0.cha  cookie_control/137-2.cha  cookie_control/227-1.cha  cookie_control/668-0.cha
cookie_control/071-1.cha  cookie_control/137-3.cha  cookie_control/229-1.cha  cookie_control/678-0.cha
cookie_control/071-2.cha  cookie_control/138-1.cha  cookie_control/229-2.cha  cookie_control/684-0.cha
cookie_control/071-3.cha  cookie_control/138-3.cha  cookie_control/232-0.cha  cookie_control/686-0.cha
cookie_control/071-4.cha  cookie_control/139-0.cha  cookie_control/232-1.cha  cookie_control/688-0.cha
cookie_control/073-0.cha  cookie_control/139-1.cha  cookie_control/242-0.cha  cookie_control/691-0.cha
cookie_control/073-1.cha  cookie_control/139-3.cha  cookie_control/242-1.cha  cookie_control/709-0.cha
cookie_control/073-3.cha  cookie_control/140-0.cha  cookie_control/242-2.cha  cookie_control/709-2.cha
cookie_control/086-0.cha  cookie_control/140-3.cha  cookie_control/243-0.cha"""
    files = con_files.split()
    cons = []
    for file in files:
        cha_file = open(file, 'r')
        lines = cha_file.readlines()
        
        pat_lines = []
        for i in range(len(lines)):
            if lines[i][0:5] == "*PAR:" or (lines[i][0:1] == "\t" and lines[i-1][0:5] == "*PAR:"):
                line = lines[i].split()
                if line[0] == "*PAR:":
                    
                    if len(line[-1]) > 7 and (line[-1][4] in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]):
                        new_line = line[1:-1]
                    else:
                        new_line = line[1:]
                else:
                    new_line = line[:-1]
                end_line = []
                for word in new_line:
                    if word !="[+":
                        end_line += split_word(word)

                pat_lines +=end_line

        cons.append(pat_lines)
        cha_file.close()
    return cons

def split_word(word):
    split_list = []
    end_car = False
    if word[0] == "<":
        split_list.append("<")
        word = word[1:]
        if len(word) == 0:
            return split_list
    if word[0] == "&":
        if (word != "&um") and (word != "&uh"):
            split_list.append("&")
            word = word[1:]
        if len(word) == 0:
            return split_list
    if word[-1] == ">":
        word = word[:-1]
        end_car = True
    
    if word[-5:] == "in(g)":
        word = word[:-5] + "ing"

    # if word[-2:] == "'s":
    #     split_list.append(word[:-2])
    #     word = word[:-2]
    #     split_list.append("'s")
    # elif word[-3:] == "ing":
    #     if not word in ["thing", "sing", "ring", "ping", "ding", "king", "wring"]:
    #         split_list.append(word[:-3])
    #         split_list.append("_ing")
    # elif word[-1] == "s":
    #     split_list.append(word[:-1])
    #     split_list.append("_s")

    split_list.append(word)
    if end_car:
        split_list.append(">")

    return split_list


def main():
    
    dems = read_dem_files()
    cons = read_con_files()
    flat_list = [item for sublist in dems for item in sublist]

    print(len(set(flat_list)))

    dems_train = dems[0:263]
    dems_test = dems[263:]
    cons_train = cons[0:206]
    cons_test = cons[206:]

    num_words = 1780
    oov_token = '<UNK>'
    pad_type = 'post'
    trunc_type = 'post'

    # Tokenize our training data
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token=oov_token, filters="_")
    tokenizer.fit_on_texts(dems_train)

    # Get our training data word index
    word_index = tokenizer.word_index

    # Encode training data sentences into sequences
    train_sequences = tokenizer.texts_to_sequences(dems_train)
    train_sequences2 = tokenizer.texts_to_sequences(cons_train)

    # Get max training sequence length
    maxlen = max([len(x) for x in train_sequences])
    maxlen2 = max([len(x) for x in train_sequences2])
    maxlen = max(maxlen, maxlen2)

    # Pad the training sequences
    dems_train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen, value=-1)

    test_sequences = tokenizer.texts_to_sequences(dems_test)
    dems_test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen, value=-1)
    # print(tokenizer.sequences_to_texts(dems_test_padded))
    # print(dems_test_padded)

    tokenizer.fit_on_texts(cons_train)

    # Get our training data word index
    word_index = tokenizer.word_index

    # Get max training sequence length
    maxlen = max([len(x) for x in train_sequences2])

    # Pad the training sequences
    cons_train_padded = pad_sequences(train_sequences2, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

    test_sequences = tokenizer.texts_to_sequences(cons_test)
    cons_test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

    print(tokenizer.get_config())
    return (dems_train_padded, dems_test_padded, cons_train_padded, cons_test_padded)



if __name__ == "__main__":
    main()






"""
NOTES
remove \ and _ and : and ()s and tokens at end of pat lines
add pad token to dict?
add labels
"""