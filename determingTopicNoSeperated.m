set(0,'DefaultFigureVisible','off')

%% Use a file datastore to convert txt files into strings

fds= fileDatastore("ALLTXTSoFar/*.txt","ReadFcn",@extractFileText)

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
documents = preprocessText(strArray);

%% Make result folder
mkdir("seperate_perplexity25Feb")
cd seperate_perplexity25Feb/


for NGramNo = 1:3

    switch NGramNo
        case 1
            mkdir("UnigramLDAResults_25Feb")
            cd UnigramLDAResults_25Feb/
            NameStarterGram="Unigram_"

        case 2
            %% Make Bigram result folder
            mkdir("BigramLDAResults_25Feb")
            cd BigramLDAResults_25Feb/
            NameStarterGram="Bigram_"

        case 3
            %% Make Trigram result folder
            mkdir("TrigramLDAResults_25Feb")
            cd TrigramLDAResults_25Feb/
            NameStarterGram="Trigram_"
    end

     %% Build Ngram word bag
        bag = bagOfNgrams(documents,"NgramLengths",NGramNo)
        bag = removeInfrequentNgrams(bag, 10, 'NgramLengths',NGramNo);
        bag = removeEmptyDocuments(bag);

    %% Split 10% of data for cross-validation
    numDocuments = numel(documents);

   for fold = 0.1:0.1:0.5
%     for fold = 0.1:0.1:0.2


        cvp = cvpartition(numDocuments,'HoldOut',fold);
        documentsTrain = documents(cvp.training);
        documentsValidation = documents(cvp.test);

        numTopicsRange = [5 10 15 20 25 30 35 40 ];
%         numTopicsRange = [5 10 15];

        for i = 1:numel(numTopicsRange)
            numTopics = numTopicsRange(i);

            mdl = fitlda(bag,numTopics, ...
                'Solver','savb', ...
                'Verbose',0);

            [~,validationPerplexity(i)] = logp(mdl,documentsValidation);
            timeElapsed(i) = mdl.FitInfo.History.TimeSinceStart(end);
        end
     

        figure
        yyaxis left
        plot(numTopicsRange,validationPerplexity,'+-')
        ylabel("Validation Perplexity")

        yyaxis right
        plot(numTopicsRange,timeElapsed,'o-')
        ylabel("Time Elapsed (s)")

        legend(["Validation Perplexity" "Time Elapsed (s)"],'Location','southeast')
        xlabel("Number of Topics")
        foldname=num2str(fold)
        Name = NameStarterGram+"_"+foldname+"_"+"Perplexity.pdf"
        saveas(gcf,Name)

        tablename= NameStarterGram+"_"+foldname+"_"+"Perplexity.csv"
        writematrix(validationPerplexity,tablename)

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
