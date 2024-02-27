
set(0,'DefaultFigureVisible','off')

%% Use a file datastore to convert txt files into strings
fds= fileDatastore("fullTXT/*.txt","ReadFcn",@extractFileText)

unwanted = '1234567890';
strArray = [];
while hasdata(fds)
    str = read(fds);
    str = convertStringsToChars(str);
    str = str(~ismember(str, unwanted));
    str = convertCharsToStrings(str);
    strArray = [strArray; str];
end
reset(fds);

%% Preprocess on the string array
documentsFDS = preprocessText(strArray);

%% Uni and Bigram
for NGramNo = 1:3

    switch NGramNo
        case 1
            mkdir("UnigramLDAResults_24Feb")
            cd UnigramLDAResults_24Feb/
            NameStarterGram="Unigram_"

        case 2
            %% Make Bigram result folder
            mkdir("BigramLDAResults_24Feb")
            cd BigramLDAResults_24Feb/
            NameStarterGram="Bigram_"

        case 3
            %% Make Trigram result folder
            mkdir("TrigramLDAResults_24Feb")
            cd TrigramLDAResults_24Feb/
            NameStarterGram="Trigram_"
    end

    %% Build Ngram word bag
    bagFDS = bagOfNgrams(documentsFDS,"NgramLengths",NGramNo)
    bagFDS = removeInfrequentNgrams(bagFDS, 10, 'NgramLengths',NGramNo);

    bagFDS = removeEmptyDocuments(bagFDS);
    save("preprocessed","bagFDS")


    %% Loop topic number [5:5:40]
    for numTopicsFDS = 5:5:40

        NameStarter=NameStarterGram+num2str(numTopicsFDS)

        %% LDA modelling
        rng("default")
        mdlFDS = fitlda(bagFDS,numTopicsFDS,Verbose=0);
        save("LDAmodel" + numTopicsFDS, "mdlFDS", "numTopicsFDS")


        %% Get Top words of each topic and the score
        TopWordsEachTopic= []
        for i = 1:numTopicsFDS
            TopWordsEachTopic=[TopWordsEachTopic;["Topic",i]]

            %% customize the number of top words in each topic
            topFDS = topkwords(mdlFDS,3,i);
            top10FDS = topkwords(mdlFDS,10,i);

            topFDSString = table2array(top10FDS)
            TopWordsEachTopic = [TopWordsEachTopic;topFDSString]
            TopWordsEachTopic = [TopWordsEachTopic; ["-","-"]]
            topWordsFDS(i) = join(topFDS.Word,", ");

        end

        %% Output - Top words of each topic
        TopWordsName = NameStarter+"_"+"TopWords.csv"
        writematrix(TopWordsEachTopic,TopWordsName)

        %% find and locate representative papers of each topic
        reprensentativeEachTopicTable=[]
        for i = 1:numTopicsFDS
            reprensentativeEachTopicTable=[reprensentativeEachTopicTable;["Topic",i," "," "]]

            %% get each column (i.e., each paper's probability under one topic)
            column = mdlFDS.DocumentTopicProbabilities( :, i);
            %% Get top n value in each column and index their rows using maxk(column, n)
            [TopEachColumn,Index]=maxk(column,10);
            %% Generate a same size empty string as strArray
            emptystrArray = repmat("",size(strArray))
            %% using the index to find the full text paper
            %% Get the exact file
            filepath=fds.Files(Index)
            %% Use fileparts to split the path into parts
            [~,name,ext] = fileparts(filepath);
            %% Concatenate name and ext to get only the file name
            filename = [name];

            TopEachColumnArrays=[TopEachColumn,Index,filename,emptystrArray(Index)]
            %% add sperator to make the table readable
            AddSeperator=[TopEachColumnArrays;["-","-","-","-"]];
            %% combine each column's result into a whole table
            reprensentativeEachTopicTable=[reprensentativeEachTopicTable;AddSeperator]    %% use rows to get the text of paper

        end
        %% Output - Representative Paper
        RepresentativeName = NameStarter+"_"+"Representative.csv"

        writematrix(reprensentativeEachTopicTable,RepresentativeName)

        %% Portion of each topic in all database
        %% Mapping the model with the whole document set
        topicMixturesFDS = transform(mdlFDS,documentsFDS);

        figure
        bar(topicMixturesFDS(1,:))

        xlabel("Topic")
        xticklabels(topWordsFDS);
        ylabel("Probability")
        title("Document Topic Probabilities")

        %% Output - PortionOfTopic
        PortionName = NameStarter+"_"+"Portion.pdf"
        saveas(gcf,PortionName)

        %% Mapping each article to the topics

        figure
        barh(topicMixturesFDS,"stacked")
        xlim([0 1])

        title("Topic Mixtures")
        xlabel("Topic Probability")
        ylabel("Document")

        legend(topWordsFDS, ...
            Location="southoutside", ...
            NumColumns=2)

        %% Output - Mapping each article to themes
        MappingName = NameStarter+"_"+"Mapping.pdf"
        saveas(gcf,MappingName)

        %% For each topic, find articles where it has the highest probablity over other topics
        %% Articles where blue is the longest than other colours

        varTypes = ["double","double","string","double"];
        varNames=["A","Topic","name","probability"];
        sz =[4 4];

        BlueLongerThanOtherColour=table('size',sz,'VariableTypes',varTypes,'VariableNames',varNames);
        HowManyArticles = height(topicMixturesFDS)
        for articleNo = 1:HowManyArticles
            scanrow = topicMixturesFDS(articleNo,:)
            [M,BlueTopic]=max(scanrow(:))
            stringarray=fds.Files(articleNo)
            [~,name1,ext] = fileparts(stringarray);
            %% Concatenate name and ext to get only the file name
            filename1 = [name1];
            filename2= convertCharsToStrings(filename1)
            BlueLongerThanOtherColour(articleNo,:)={articleNo BlueTopic filename2 M}
        end

      %% Categorize articles based on the topics
    T = BlueLongerThanOtherColour
    [G,ID] = findgroups(T.Topic);

    All_A = splitapply(@(x) {x}, T.name, G);
    All_B= splitapply(@(x) {x}, T.probability, G);
    TTA = table(ID, All_A,'VariableNames',{'Topic','Article_Name'});
    TTB = table(ID, All_B,'VariableNames',{'Topic','Probability'});

    BluestA = NameStarter+"_"+"Bluest_PaperName.csv"
    BluestB = NameStarter+"_"+"Bluest_PaperProbability.csv"
    BluestC = NameStarter+"_"+"Bluest_PaperCount.csv"
    writetable(TTA,BluestA)
    writetable(TTB,BluestB)

    % Assuming your table is stored in a variable called 'table'
    % and the column with paper numbers is called 'paper_number'
    % and the column with topics is called 'Topic'

    % Count the number of papers for each topic
    num_topics = numTopicsFDS;
    num_papers_per_topic = zeros(num_topics, 1);
    for i = 1:num_topics
        num_papers_per_topic(i) = sum(T.Topic == i);
    end

    % Find out which topics have no papers
    topics_with_no_papers = find(num_papers_per_topic == 0);

    % Display the results
    for i = 1:num_topics
        fprintf('Topic %d has %d papers\n', i, num_papers_per_topic(i));
    end
    fprintf('Topics with no papers: %s\n', mat2str(topics_with_no_papers));

    % Create a table with the results
    BluestPapersCount = table((1:num_topics)', num_papers_per_topic, 'VariableNames', {'Topic', 'pcs'});
    writetable(BluestPapersCount,BluestC)
    

    %% Word Cloud Themes
    figure
    t = tiledlayout("flow");
    title(t,"LDA Topics")

    for i = 1:numTopicsFDS
        nexttile
        wordcloud(mdlFDS,i,MaxDisplayWords=10);
        title({"Topic " + i});
        %% ,BluestPapersCount.pcs(i)+ " pcs"
    end

    %% Output - Word Cloud
    WordCloudName = NameStarter+"_"+"WordCloud.pdf"
    saveas(gcf,WordCloudName)


    end


    %% return to father folder
    cd('..');
end

%% Preprocessing
function documents = preprocessText(textData)

% Tokenize the text.
documents = tokenizedDocument(textData);

% Lemmatize the words.
documents = addPartOfSpeechDetails(documents);
documents = normalizeWords(documents,Style="lemma");

% Erase punctuation.
documents = erasePunctuation(documents);

% Remove a list of stop words, including customs ones. should also remove
% numbers
customStopWords = [stopWords "ieee" "licence" "wiley" "university" "kong" "hong" "proceedings" "fig" "acm"];
documents = removeWords(documents, customStopWords);

% Remove words with 2 or fewer characters, and words with 15 or greater
% characters.
documents = removeShortWords(documents,2);
documents = removeLongWords(documents,15);

end