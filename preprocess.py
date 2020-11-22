# file for preprocessing


def read_files():


    cha_file = open('cookie_dem/001-0.cha', 'r')
    lines = cha_file.readlines()
    pat_lines = []
    for line in lines:
        line = line.split()
        if line[0] == "*PAR:":
            pat_lines.append(line[1:-1])

    print(pat_lines)



def main():
    read_files()


if __name__ == "__main__":
    main()