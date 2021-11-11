from io import StringIO, SEEK_CUR, SEEK_END, SEEK_SET


class ReadAppend_StringIO(StringIO):

    def append(self, s):
        current_pos = self.seek(0, SEEK_CUR)
        end_of_buffer = self.seek(0, SEEK_END)
        result = self.write(s)
        self.seek(current_pos, SEEK_SET)
        return result

if __name__ == '__main__':
    s = ReadAppend_StringIO()
    print(1)
    s.append('Hola que tal\n')
    print(2)
    s.append('Estupendamente\n')
    print(3)
    print(s.readline(), end='')
    print(4)
    s.append('Nos vemos\n')
    print(5)
    print(s.readline(), end='')
    print(6)
    print(s.readline(), end='')
    print(7)
    s.append('Hasta mañana\n')
    print(8)
    print(s.readline(), end='')
    print(9)
    print(s.getvalue())