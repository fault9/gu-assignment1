# Example tokenizations

en (valid line 1) — 108 characters

After these victories, he was appointed as the organizing secretary of DMK in southern Tamil Nadu districts.

**char** — 109 tokens, 0.99 chars/token

```
▁|A|f|t|e|r|▁|t|h|e|s|e|▁|v|i|c|t|o|r|i|e|s|,|▁|h|e|▁|w|a|s|▁|a|p|p|o|i|n|t|e|d|▁|a|s|▁|t|h|e|▁|o|r|g|a|n|i|z|i|n|g|▁|s|e|c|r|e|t|a|r|y|▁|o|f|▁|D|M|K|▁|i|n|▁|s|o|u|t|h|e|r|n|▁|T|a|m|i|l|▁|N|a|d|u|▁|d|i|s|t|r|i|c|t|s|.
```

**bpe_2k** — 53 tokens, 2.04 chars/token

```
▁A|f|ter|▁the|se|▁v|ic|t|or|ies|,|▁he|▁was|▁a|p|p|o|in|ted|▁as|▁the|▁or|g|an|iz|ing|▁s|ec|r|et|ary|▁of|▁D|M|K|▁in|▁s|out|her|n|▁T|am|il|▁N|ad|u|▁d|ist|r|ic|t|s|.
```

**bpe_10k** — 33 tokens, 3.27 chars/token

```
▁After|▁these|▁v|ict|ories|,|▁he|▁was|▁app|oin|ted|▁as|▁the|▁organiz|ing|▁sec|ret|ary|▁of|▁D|M|K|▁in|▁s|outhern|▁Tam|il|▁N|adu|▁dist|ric|ts|.
```

tr (valid line 13) — 93 characters

Demiyelinize edici kalıbının yerine demiyelinizan sözcüğünün kullanıldığına da rastlanabilir.

**char** — 94 tokens, 0.99 chars/token

```
▁|D|e|m|i|y|e|l|i|n|i|z|e|▁|e|d|i|c|i|▁|k|a|l|ı|b|ı|n|ı|n|▁|y|e|r|i|n|e|▁|d|e|m|i|y|e|l|i|n|i|z|a|n|▁|s|ö|z|c|ü|ğ|ü|n|ü|n|▁|k|u|l|l|a|n|ı|l|d|ı|ğ|ı|n|a|▁|d|a|▁|r|a|s|t|l|a|n|a|b|i|l|i|r|.
```

**bpe_2k** — 45 tokens, 2.07 chars/token

```
▁D|em|iy|el|in|iz|e|▁ed|ic|i|▁k|al|ı|b|ının|▁yer|ine|▁d|em|iy|el|in|iz|an|▁s|ö|z|c|ü|ğ|ün|ün|▁kul|lan|ıl|dı|ğ|ına|▁da|▁r|ast|lan|ab|ilir|.
```

**bpe_10k** — 31 tokens, 3.00 chars/token

```
▁Dem|iy|el|in|ize|▁ed|ici|▁kal|ı|b|ının|▁yerine|▁dem|iy|el|in|iz|an|▁söz|cü|ğ|ünün|▁kullan|ıldı|ğ|ına|▁da|▁rast|lan|abilir|.
```

zh (valid line 3) — 59 characters

玻利维亚有一名运动员获得了IJF给予的一个额外美洲席位，该国亦时隔12年再度在奥运柔道赛场上亮相 citation 。

**char** — 60 tokens, 0.98 chars/token

```
▁|玻|利|维|亚|有|一|名|运|动|员|获|得|了|I|J|F|给|予|的|一|个|额|外|美|洲|席|位|，|该|国|亦|时|隔|1|2|年|再|度|在|奥|运|柔|道|赛|场|上|亮|相|▁|c|i|t|a|t|i|o|n|▁|。
```

**bpe_2k** — 53 tokens, 1.11 chars/token

```
▁|<0xE7>|<0x8E>|<0xBB>|利|维|亚|有|一|名|运|动|员|获|得|了|I|J|F|给|予|的|一个|额|外|美|洲|席|位|，|该|国|亦|时|隔|12|年|再|度|在|奥|运|<0xE6>|<0x9F>|<0x94>|道|赛|场|上|亮|相|▁citation|▁。
```

**bpe_10k** — 44 tokens, 1.34 chars/token

```
▁|<0xE7>|<0x8E>|<0xBB>|利|维|亚|有一|名|运动员|获得了|I|J|F|给|予|的一个|额|外|美洲|席|位|，|该|国|亦|时|隔|12|年|再度|在|奥运|<0xE6>|<0x9F>|<0x94>|道|赛|场|上|亮|相|▁citation|▁。
```

