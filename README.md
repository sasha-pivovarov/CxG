# Выбор предлога в английских глаголах эмоционального состояния

### Материалы
<a href="github.com/sasha-pivovarov/CxG/tagged2.csv> Данные </a>

## Рабочая гипотеза

Для конструкций с глаголами эмоционального состояния типа agonize, grieve, worry, fret, obsess выбор предлога, вводящего тему (over или about) зависит либо от характеристик темы, либо от положения глагола в структуре зависимостей. Также оно может зависеть от самого глагола.

### Материал исследования
781 пример из корпуса Araneum Anglicum Maximum, размеченные при помощи парсера зависимостей spaCy. Всего шесть признаков: строка глагола, положение глагола в структуре зависимстей, часть речи темы (fine-grained), часть речи темы (coarse), строка темы, положение темы в структуре зависимостей.

Названия переменных:
DEP_VAL: положение темы в структуре зависимостей
DEP_TAG: часть речи темы (fine-grained)
DEP_STR: лемма темы
DEP_POS: часть речи темы (coarse)
VERB_STR: лемма глагола
VERB_DEP: положение глагола в структуре зависимостей

## Анализ: дескриптивная статистика
Целевая переменная - строка предлога (over или about). Хи-квадрат и ANOVA для признаков:

X1        |  chi2              |  chi2_pval
----------|--------------------|----------------------
DEP_VAL   |  2.11720932984933  |  0.145651726443201
DEP_TAG   |  18.598940165367   |  1.61310204685612e-05
DEP_STR   |  43.1405577209725  |  5.09451653998692e-11
DEP_POS   |  23.3862564036554  |  1.32521976743555e-06
VERB_STR  |  9.03706254192956  |  0.00264560396102244
VERB_DEP  |  15.7814510289386  |  7.10961227397605e-05

Здесь мы можем видеть, что положение темы в структуре зависимостей и лемма глагола не являются независимыми от остальных признаков.

![alt text](https://i.imgur.com/Y4bi3tZ.png "Verb dependency structure tag")
Распределение всех положений глагола в структуре зависимостей между over и about.
![alt text](https://i.imgur.com/MGH32Nu.png "Theme dependency structure tag")
Распределение всех положений темы в структуре зависимостей между over и about.
![alt text](https://i.imgur.com/Ysq6ab0.png "Theme POS tag (fine-grained)")
Распределение всех частей речи в структуре зависимостей между over и about.
![alt text](https://i.imgur.com/N030wFz.png "Theme POS tag (coarse)")
Распределение всех частей речи (coarse) в структуре зависимостей между over и about.

## Мультифакторный анализ
Далее использовался лес решений, чтобы оценить важность признаков, после чего было обучено дерево решений. Результаты классификации на обучающей и тестовой выборке:

TRAIN:
             precision    recall  f1-score   support

      about       0.96      0.98      0.97       358
       over       0.98      0.96      0.97       424

avg / total       0.97      0.97      0.97       782

TEST:
             precision    recall  f1-score   support

      about       0.56      0.57      0.57        91
       over       0.62      0.61      0.62       105

    avg / total       0.59      0.59      0.59       196

ACCURACY: 0.581632653061

После устранения переобучения (ограничением количества листьев и максимальной глубины дерева) результат улучшился до следующего:

             precision    recall  f1-score   support

      about       0.76      0.40      0.52        86
       over       0.66      0.90      0.76       110

    avg/total       0.70      0.68      0.65       196

ACCURACY: 0.678571428571

Файлы tree.pdf и tree2.pdf - визуализации деревьев до и после устранения переобучения. 
Результаты отражают некоторый перевес в сторону over (низкий recall about в отличие от over указывает на то, что модель предпочитает ошибаться в сторону over, более частотного в корпусе).
Если использовать только категориальные признаки с малым количеством категорий (не использовать леммы) то в принципе тоже можно получить сравнимую точность, что показывает, что какая-то корреляция с грамматическими признаками есть, хотя их и недостаточно:

             precision    recall  f1-score   support

      about       0.71      0.43      0.54        90
       over       0.64      0.85      0.73       106

    avg / total       0.67      0.66      0.64       196
    
Дерево для этого эксперимента показано в tree3.pdf
Однако деревья решений scikit-learn не похволяют моделировать категориальные признаки (так или иначе они интерпретируются как ординальные), поэтому также была обучена модель в R (party).
![alt text](https://i.imgur.com/6egbA3n.png "R tree graph")
Variables actually used in tree construction:
[1] DEP_TAG  VERB_DEP VERB_STR

Root node error: 348/763 = 0.45609

n= 763 

        CP nsplit rel error  xerror     xstd
1 0.181034      0   1.00000 1.00000 0.039534
2 0.015086      1   0.81897 0.85345 0.038702
3 0.012931      5   0.75862 0.87644 0.038881
4 0.010000      7   0.73276 0.88218 0.038923

Эта модель имеет сравнительно большую ошибку, но вероятно, дело в sklearn.

## Содержательный лингвистический анализ результатов статистического анализа
![alt text](https://i.imgur.com/JQfQcL2.png "Feature importances in the same order as on table")
#
В итоге оказалось, что строка темы имеет большую значимость, чем ожидалось изначально, а её частеречные характеристики не играют почти никакой роли. Это не совсем понятно, так как кажется, что есть различие по приемлемости между he worried about doing smth., и (?)he worried over doing smth., тогда как между he worried about John's career и he worried over John's career таковых вроде бы не наблюдается. Положение глагола в структуре зависимостей (Root VP vs. clausal VP) и строка глагола (их пять) имеют некоторую значимость, хоть и меньшую, чем строка темы. Это наводит на мысль, что либо среди этих переменных нет истинного виновника всего этого различия, либо что предлог подбирается говорящим по биграмматичности с темой.  

## Обсуждение использованных квантитативных методов
Вторая мысль кажется здравой в свете результатов классификации: категорий для строк темы довольно много, и в датасете довольно низкий саппорт для каждой из них. Это могло бы объяснить сравнительно небольшую (хотя и значительно выше случайной) точность.
