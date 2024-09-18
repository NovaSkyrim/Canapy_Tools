def split_train_test(corpus, *, redo=False, **kwargs):
    """Build train and test sets from data for syntactic training with predefined test set.
    Ensure that at least one example of each syllable is present in the train set.
    
    Parameters:
    corpus: object containing dataset and configurations.
    redo: If True, forces a re-split even if 'train' column already exists.
    train_set: Optional. Predefined training set as a DataFrame or a list of sequence IDs.
    test_set: Optional. Predefined test set as a DataFrame or a list of sequence IDs.
    """
    df = corpus.dataset

    # Already split!
    if "train" in df and not redo:
        return corpus

    config = corpus.config.transforms.training

    rs = np.random.default_rng(corpus.config.misc.seed)

    df["seqid"] = df["annotation"].astype(str) + df["sequence"].astype(str)

    n_sequences = len((df["seqid"]).unique())

    max_sequences = config.max_sequences
    if max_sequences == -1:
        max_sequences = n_sequences

    if max_sequences > n_sequences:
        raise ValueError(
            f"Can't select {max_sequences} training sequences "
            f"in a dataset of {n_sequences} sequences."
        )

    # Train dataframe
    train_df = pd.DataFrame(columns=df.columns)

    if corpus.train_set is not None:
        if isinstance(corpus.train_set, pd.DataFrame):
            train_seqs = corpus.train_set["seqid"].unique()
        elif isinstance(corpus.train_set, list):
            train_seqs = corpus.train_set
        else:
            raise ValueError("corpus.train_set should be either a DataFrame or a list of sequence IDs.")
        
        train_df = df.query("seqid in @train_seqs")
        already_picked = train_df["seqid"].unique()
    else:
        # Default behavior: ensure one example of each class
        class_min_occurence = df.groupby("label")["label"].count().min()
        n_classes = len(df["label"].unique())
        while len(train_df.groupby("label")) < n_classes:
            class_min_occurence += 1
            min_occurences = (
                df.groupby("label")["label"]
                .count()
                .index[df.groupby("label")["label"].count() < class_min_occurence]
            )
            min_occurences_seqs = df.query("label in @min_occurences")["seqid"].unique()
            train_df = df.query("seqid in @min_occurences_seqs")

        already_picked = train_df["seqid"].unique()

    left_to_pick = df.query("seqid not in @already_picked")["seqid"].unique()

    logger.info(
        f"Min. number of sequences to train over all classes: "
        f"{len(already_picked)}"
    )

    if max_sequences < len(already_picked):
        logger.warning(
            f"Only {max_sequences} sequences will be selected (from max_sequence "
            f"config parameter), but {len(already_picked)} sequences are necessary "
            f"to train over all existing label classes."
        )

    # Handle predefined test set
    if corpus.test_set is not None:
        if isinstance(corpus.test_set, pd.DataFrame):
            test_seqs = corpus.test_set["seqid"].unique()
        elif isinstance(corpus.test_set, list):
            test_seqs = corpus.test_set
        else:
            raise ValueError("test_set should be either a DataFrame or a list of sequence IDs.")
        
        test_df = df.query("seqid in @test_seqs")
        left_to_pick = df.query("seqid not in @test_seqs and seqid not in @already_picked")["seqid"].unique()
    else:
        # Add data to train_df up to test_ratio if no test set is provided
        test_ratio = config.test_ratio

        more_size = np.floor(
            (1 - test_ratio) * len(left_to_pick) - test_ratio * len(already_picked)
        ).astype(int)

        some_more_seqs = rs.choice(left_to_pick, size=more_size, replace=False)
        some_more_data = df.query("seqid in @some_more_seqs")

        # Split
        train_df = pd.concat([train_df, some_more_data])
        test_df = df.query("seqid not in @train_df.seqid.unique()")

    # Reduce sequence number up to max_sequences
    n_train_seqs = len(train_df["seqid"].unique())
    if max_sequences < n_train_seqs:
        sequences = train_df["seqid"].unique()
        selection = rs.choice(sequences, size=max_sequences, replace=False)
        train_df = train_df.query("seqid in @selection")

    df["train"] = False
    df.loc[train_df.index, "train"] = True

    # Time stats
    train_time = (train_df["offset_s"] - train_df["onset_s"]).sum()
    test_time = (test_df["offset_s"] - test_df["onset_s"]).sum()

    silence_tag = corpus.config.transforms.annots.silence_tag
    train_no_silence = train_df.query("label != @silence_tag")
    test_no_silence = test_df.query("label != @silence_tag")

    train_nosilence_time = (
        train_no_silence["offset_s"] - train_no_silence["onset_s"]
    ).sum()
    test_nosilence_time = (
        test_no_silence["offset_s"] - test_no_silence["onset_s"]
    ).sum()

    logger.info(
        f"Final repartition of data - "
        f"\nTrain: {len(train_df['seqid'].unique())} ({len(train_df)} labels "
        f"- {train_time:.3f} s - {train_nosilence_time:.3f} s (w/o silence)"
        f"\nTest: {len(test_df['seqid'].unique())} ({len(test_df)} labels) "
        f"- {test_time:.3f} s - {test_nosilence_time:.3f} s (w/o silence)"
    )

    df.drop("seqid", axis=1, inplace=True)

    train_df.to_csv(f"D:/Inria/Experiments/DATASIZE/Tests_datasize/3_1/train_set.csv", index=False)
    test_df.to_csv(f"D:/Inria/Experiments/DATASIZE/Tests_datasize/3_1/test_set.csv", index=False)

    return corpus
