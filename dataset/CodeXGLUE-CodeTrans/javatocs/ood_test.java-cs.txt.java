public CreateVariableResult createVariable(CreateVariableRequest request) {request = beforeClientExecution(request);return executeCreateVariable(request);}
public ValueEval evaluate(int srcRowIndex, int srcColumnIndex, ValueEval arg0, ValueEval arg1) {try {AreaEval reA = evaluateRef(arg0);AreaEval reB = evaluateRef(arg1);AreaEval result = resolveRange(reA, reB);if (result == null) {return ErrorEval.NULL_INTERSECTION;}return result;} catch (EvaluationException e) {return e.getErrorEval();}}
final public SrndQuery PrimaryQuery() throws ParseException {SrndQuery q;switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {case LPAREN:jj_consume_token(LPAREN);q = FieldsQuery();jj_consume_token(RPAREN);break;case OR:case AND:case W:case N:q = PrefixOperatorQuery();break;case TRUNCQUOTED:case QUOTED:case SUFFIXTERM:case TRUNCTERM:case TERM:q = SimpleTerm();break;default:jj_la1[5] = jj_gen;jj_consume_token(-1);throw new ParseException();}OptionalWeights(q);{if (true) return q;}throw new Error("Missing return statement in function");}
public ObjectId getObjectId() {return objectId;}
public CreateOrUpdateTagsResult createOrUpdateTags(CreateOrUpdateTagsRequest request) {request = beforeClientExecution(request);return executeCreateOrUpdateTags(request);}
public GetFieldLevelEncryptionConfigResult getFieldLevelEncryptionConfig(GetFieldLevelEncryptionConfigRequest request) {request = beforeClientExecution(request);return executeGetFieldLevelEncryptionConfig(request);}
public void run() {doRefreshProgress(mId, mProgress, mFromUser, true);mRefreshProgressRunnable = this;}
public static final int match(byte[] b, int ptr, byte[] src) {if (ptr + src.length > b.length)return -1;for (int i = 0; i < src.length; i++, ptr++)if (b[ptr] != src[i])return -1;return ptr;}
public QueryConfigHandler getQueryConfigHandler() {return this.queryConfig;}
public CreateDataSourceFromRedshiftResult createDataSourceFromRedshift(CreateDataSourceFromRedshiftRequest request) {request = beforeClientExecution(request);return executeCreateDataSourceFromRedshift(request);}
public DeleteApnsVoipChannelResult deleteApnsVoipChannel(DeleteApnsVoipChannelRequest request) {request = beforeClientExecution(request);return executeDeleteApnsVoipChannel(request);}
public ListHyperParameterTuningJobsResult listHyperParameterTuningJobs(ListHyperParameterTuningJobsRequest request) {request = beforeClientExecution(request);return executeListHyperParameterTuningJobs(request);}
public void write(int oneChar) {doWrite(new char[] { (char) oneChar }, 0, 1);}
public ActivateUserResult activateUser(ActivateUserRequest request) {request = beforeClientExecution(request);return executeActivateUser(request);}
public String getClassName() {return className;}
public CreateStreamingDistributionResult createStreamingDistribution(CreateStreamingDistributionRequest request) {request = beforeClientExecution(request);return executeCreateStreamingDistribution(request);}
public SpanQuery makeSpanClause() {SpanQuery [] spanQueries = new SpanQuery[size()];Iterator<SpanQuery> sqi = weightBySpanQuery.keySet().iterator();int i = 0;while (sqi.hasNext()) {SpanQuery sq = sqi.next();float boost = weightBySpanQuery.get(sq);if (boost != 1f) {sq = new SpanBoostQuery(sq, boost);}spanQueries[i++] = sq;}if (spanQueries.length == 1)return spanQueries[0];else return new SpanOrQuery(spanQueries);}
public int findEndOffset(StringBuilder buffer, int start) {if( start > buffer.length() || start < 0 ) return start;bi.setText(buffer.substring(start));return bi.next() + start;}
public EscherGraphics(HSSFShapeGroup escherGroup, HSSFWorkbook workbook, Color forecolor, float verticalPointsPerPixel ){this.escherGroup = escherGroup;this.workbook = workbook;this.verticalPointsPerPixel = verticalPointsPerPixel;this.verticalPixelsPerPoint = 1 / verticalPointsPerPixel;this.font = new Font("Arial", 0, 10);this.foreground = forecolor;}
public QueryParserTokenManager(CharStream stream, int lexState){this(stream);SwitchTo(lexState);}
public FST.BytesReader getBytesReader() {return fst.getBytesReader();}
public TokenFilter create(TokenStream input) {CommonGramsFilter commonGrams = (CommonGramsFilter) super.create(input);return new CommonGramsQueryFilter(commonGrams);}
public BytesRef next() {termUpto++;if (termUpto >= info.terms.size()) {return null;} else {info.terms.get(info.sortedTerms[termUpto], br);return br;}}
public FetchPhotosRequest() {super("CloudPhoto", "2017-07-11", "FetchPhotos", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public void mode(int m) {_mode = m;}
public String getUserInfo() {return decode(userInfo);}
public boolean isSubTotal(int rowIndex, int columnIndex) {return false;}
public void setValue(boolean newValue) {value = newValue;}
public RuleTagToken(String ruleName, int bypassTokenType, String label) {if (ruleName == null || ruleName.isEmpty()) {throw new IllegalArgumentException("ruleName cannot be null or empty.");}this.ruleName = ruleName;this.bypassTokenType = bypassTokenType;this.label = label;}
public void clearDFA() {for (int d = 0; d < decisionToDFA.length; d++) {decisionToDFA[d] = new DFA(atn.getDecisionState(d), d);}}
@Override public boolean remove(Object o) {return Impl.this.remove(o) != null;}
public SubmoduleInitCommand(Repository repo) {super(repo);paths = new ArrayList<>();}
public String toString(){StringBuilder buffer = new StringBuilder();buffer.append( "[RightMargin]\n" );buffer.append( "    .margin               = " ).append( " (" ).append( getMargin() ).append( " )\n" );buffer.append( "[/RightMargin]\n" );return buffer.toString();}
public CommonTokenStream(TokenSource tokenSource, int channel) {this(tokenSource);this.channel = channel;}
public static String insertId(String message, ObjectId changeId) {return insertId(message, changeId, false);}
public FormatRun(short character, short fontIndex) {this._character = character;this._fontIndex = fontIndex;}
public E previous() {if (iterator.previousIndex() >= start) {return iterator.previous();}throw new NoSuchElementException();}
public int getDefaultPort() {return -1;}
public DescribeTransitGatewayVpcAttachmentsResult describeTransitGatewayVpcAttachments(DescribeTransitGatewayVpcAttachmentsRequest request) {request = beforeClientExecution(request);return executeDescribeTransitGatewayVpcAttachments(request);}
public static BATBlock createEmptyBATBlock(final POIFSBigBlockSize bigBlockSize, boolean isXBAT) {BATBlock block = new BATBlock(bigBlockSize);if(isXBAT) {final int _entries_per_xbat_block = bigBlockSize.getXBATEntriesPerBlock();block._values[ _entries_per_xbat_block ] = POIFSConstants.END_OF_CHAIN;}return block;}
public void add(int location, E object) {if (location >= 0 && location <= size) {Link<E> link = voidLink;if (location < (size / 2)) {for (int i = 0; i <= location; i++) {link = link.next;}} else {for (int i = size; i > location; i--) {link = link.previous;}}Link<E> previous = link.previous;Link<E> newLink = new Link<E>(object, previous, link);previous.next = newLink;link.previous = newLink;size++;modCount++;} else {throw new IndexOutOfBoundsException();}}
public GetThreatIntelSetResult getThreatIntelSet(GetThreatIntelSetRequest request) {request = beforeClientExecution(request);return executeGetThreatIntelSet(request);}
public ListPhotoTagsRequest() {super("CloudPhoto", "2017-07-11", "ListPhotoTags", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public byte readByte() {return bytes[pos--];}
public static double calcDistanceFromErrPct(Shape shape, double distErrPct, SpatialContext ctx) {if (distErrPct < 0 || distErrPct > 0.5) {throw new IllegalArgumentException("distErrPct " + distErrPct + " must be between [0 to 0.5]");}if (distErrPct == 0 || shape instanceof Point) {return 0;}Rectangle bbox = shape.getBoundingBox();Point ctr = bbox.getCenter();double y = (ctr.getY() >= 0 ? bbox.getMaxY() : bbox.getMinY());double diagonalDist = ctx.getDistCalc().distance(ctr, bbox.getMaxX(), y);return diagonalDist * distErrPct;}
public ListAvailableManagementCidrRangesResult listAvailableManagementCidrRanges(ListAvailableManagementCidrRangesRequest request) {request = beforeClientExecution(request);return executeListAvailableManagementCidrRanges(request);}
public synchronized long ramBytesUsed() {long sizeInBytes = BASE_RAM_BYTES_USED + fields.size() * 2 * RamUsageEstimator.NUM_BYTES_OBJECT_REF;for(SimpleTextTerms simpleTextTerms : termsCache.values()) {sizeInBytes += (simpleTextTerms!=null) ? simpleTextTerms.ramBytesUsed() : 0;}return sizeInBytes;}
public ObjectIdSubclassMap<ObjectId> getBaseObjectIds() {if (baseObjectIds != null)return baseObjectIds;return new ObjectIdSubclassMap<>();}
public GroupQueryNode(QueryNode query) {if (query == null) {throw new QueryNodeError(new MessageImpl(QueryParserMessages.PARAMETER_VALUE_NOT_SUPPORTED, "query", "null"));}allocate();setLeaf(false);add(query);}
public CreateRepoBuildRuleRequest() {super("cr", "2016-06-07", "CreateRepoBuildRule", "cr");setUriPattern("/repos/[RepoNamespace]/[RepoName]/rules");setMethod(MethodType.PUT);}
public boolean canAppendMatch() {for (Head head : heads) {if (head != LastHead.INSTANCE) {return true;}}return false;}
public DeletePhotosRequest() {super("CloudPhoto", "2017-07-11", "DeletePhotos", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public Decompressor clone() {return new DeflateDecompressor();}
public E pollLast() {Map.Entry<E, Object> entry = backingMap.pollLastEntry();return (entry == null) ? null : entry.getKey();}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_print_gridlines);}
public DrawingManager2( EscherDggRecord dgg ) {this.dgg = dgg;}
public void unwriteProtectWorkbook() {records.remove(fileShare);records.remove(writeProtect);fileShare = null;writeProtect = null;}
public ListResourceRecordSetsResult listResourceRecordSets(ListResourceRecordSetsRequest request) {request = beforeClientExecution(request);return executeListResourceRecordSets(request);}
public ShapeFieldCacheDistanceValueSource(SpatialContext ctx,ShapeFieldCacheProvider<Point> provider, Point from, double multiplier) {this.ctx = ctx;this.from = from;this.provider = provider;this.multiplier = multiplier;}
public int following(int pos) {if (pos < text.getBeginIndex() || pos > text.getEndIndex()) {throw new IllegalArgumentException("offset out of bounds");} else if (0 == sentenceStarts.length) {text.setIndex(text.getBeginIndex());return DONE;} else if (pos >= sentenceStarts[sentenceStarts.length - 1]) {text.setIndex(text.getEndIndex());currentSentence = sentenceStarts.length - 1;return DONE;} else { currentSentence = (sentenceStarts.length - 1) / 2; moveToSentenceAt(pos, 0, sentenceStarts.length - 2);text.setIndex(sentenceStarts[++currentSentence]);return current();}}
public String toFormulaString(String[] operands) {if(space.isSet(_options)) {return operands[ 0 ];} else if (optiIf.isSet(_options)) {return toFormulaString() + "(" + operands[0] + ")";} else if (optiSkip.isSet(_options)) {return toFormulaString() + operands[0];   } else {return toFormulaString() + "(" + operands[0] + ")";}}
public StashApplyCommand stashApply() {return new StashApplyCommand(repo);}
public SubmoduleStatusType getType() {return type;}
public ArrayList() {array = EmptyArray.OBJECT;}
public String toString(){return image;}
public static void register(MergeStrategy imp) {register(imp.getName(), imp);}
public static CharBuffer wrap(char[] array, int start, int charCount) {Arrays.checkOffsetAndCount(array.length, start, charCount);CharBuffer buf = new ReadWriteCharArrayBuffer(array);buf.position = start;buf.limit = start + charCount;return buf;}
public int get(int i) {if (count <= i)throw new ArrayIndexOutOfBoundsException(i);return entries[i];}
public BufferedReader(Reader in, int size) {super(in);if (size <= 0) {throw new IllegalArgumentException("size <= 0");}this.in = in;buf = new char[size];}
public void setPasswordVerifier(int passwordVerifier) {this.passwordVerifier = passwordVerifier;}
public UpdateDetectorVersionResult updateDetectorVersion(UpdateDetectorVersionRequest request) {request = beforeClientExecution(request);return executeUpdateDetectorVersion(request);}
public ReverseStringFilter create(TokenStream in) {return new ReverseStringFilter(in);}
public GetPhoneNumberSettingsResult getPhoneNumberSettings(GetPhoneNumberSettingsRequest request) {request = beforeClientExecution(request);return executeGetPhoneNumberSettings(request);}
public String toString() {return "[Array Formula or Shared Formula]\n" + "row = " + getRow() + "\n" + "col = " + getColumn() + "\n";}
public DescribeStreamProcessorResult describeStreamProcessor(DescribeStreamProcessorRequest request) {request = beforeClientExecution(request);return executeDescribeStreamProcessor(request);}
public int lastIndexOf(String string) {return lastIndexOf(string, count);}
public int stem(char s[], int len) {if (len > 4 && s[len-1] == 's')len--;if (len > 5 &&(endsWith(s, len, "ene") ||  (endsWith(s, len, "ane") &&useNynorsk                 )))return len - 3;if (len > 4 &&(endsWith(s, len, "er") ||   endsWith(s, len, "en") ||   endsWith(s, len, "et") ||   (endsWith(s, len, "ar") &&useNynorsk                 )))return len - 2;if (len > 3)switch(s[len-1]) {case 'a':     case 'e':     return len - 1;}return len;}
public Object get(CharSequence key) {final int bucket = normalCompletion.getBucket(key);return bucket == -1 ? null : Long.valueOf(bucket);}
public MoveAddressToVpcResult moveAddressToVpc(MoveAddressToVpcRequest request) {request = beforeClientExecution(request);return executeMoveAddressToVpc(request);}
public ThreeWayMerger newMerger(Repository db) {return new InCoreMerger(db);}
public DescribeAliasResult describeAlias(DescribeAliasRequest request) {request = beforeClientExecution(request);return executeDescribeAlias(request);}
public static String toStringTree(Tree t) {return toStringTree(t, (List<String>)null);}
public ValueEval evaluate(ValueEval[] args, OperationEvaluationContext ec) {if (args.length < 1) {return ErrorEval.VALUE_INVALID;}boolean isA1style;String text;try {ValueEval ve = OperandResolver.getSingleValue(args[0], ec.getRowIndex(), ec.getColumnIndex());text = OperandResolver.coerceValueToString(ve);switch (args.length) {case 1:isA1style = true;break;case 2:isA1style = evaluateBooleanArg(args[1], ec);break;default:return ErrorEval.VALUE_INVALID;}} catch (EvaluationException e) {return e.getErrorEval();}return evaluateIndirect(ec, text, isA1style);}
public Reader create(Reader input) {return new PersianCharFilter(input);}
public static TermStats[] getHighFreqTerms(IndexReader reader, int numTerms, String field, Comparator<TermStats> comparator) throws Exception {TermStatsQueue tiq = null;if (field != null) {Terms terms = MultiTerms.getTerms(reader, field);if (terms == null) {throw new RuntimeException("field " + field + " not found");}TermsEnum termsEnum = terms.iterator();tiq = new TermStatsQueue(numTerms, comparator);tiq.fill(field, termsEnum);} else {Collection<String> fields = FieldInfos.getIndexedFields(reader);if (fields.size() == 0) {throw new RuntimeException("no fields found for this index");}tiq = new TermStatsQueue(numTerms, comparator);for (String fieldName : fields) {Terms terms = MultiTerms.getTerms(reader, fieldName);if (terms != null) {tiq.fill(fieldName, terms.iterator());}}}TermStats[] result = new TermStats[tiq.size()];int count = tiq.size() - 1;while (tiq.size() != 0) {result[count] = tiq.pop();count--;}return result;}
public DBSubnetGroup createDBSubnetGroup(CreateDBSubnetGroupRequest request) {request = beforeClientExecution(request);return executeCreateDBSubnetGroup(request);}
public DeleteHsmConfigurationResult deleteHsmConfiguration(DeleteHsmConfigurationRequest request) {request = beforeClientExecution(request);return executeDeleteHsmConfiguration(request);}
public DescribeVpcEndpointServicesResult describeVpcEndpointServices(DescribeVpcEndpointServicesRequest request) {request = beforeClientExecution(request);return executeDescribeVpcEndpointServices(request);}
@Override public boolean remove(Object object) {synchronized (mutex) {return c.remove(object);}}
public LsRemoteCommand setRemote(String remote) {checkCallable();this.remote = remote;return this;}
public DescribeClientVpnConnectionsResult describeClientVpnConnections(DescribeClientVpnConnectionsRequest request) {request = beforeClientExecution(request);return executeDescribeClientVpnConnections(request);}
public ModifyDBParameterGroupRequest(String dBParameterGroupName, java.util.List<Parameter> parameters) {setDBParameterGroupName(dBParameterGroupName);setParameters(parameters);}
public void copyRawTo(byte[] b, int o) {NB.encodeInt32(b, o, w1);NB.encodeInt32(b, o + 4, w2);NB.encodeInt32(b, o + 8, w3);NB.encodeInt32(b, o + 12, w4);NB.encodeInt32(b, o + 16, w5);}
public void processChildRecords() {convertRawBytesToEscherRecords();}
public long ramBytesUsed() {return values.ramBytesUsed()+ super.ramBytesUsed()+ Long.BYTES+ RamUsageEstimator.NUM_BYTES_OBJECT_REF;}
public int getBeginIndex() {return start;}
public static PredictionContext fromRuleContext(ATN atn, RuleContext outerContext) {if ( outerContext==null ) outerContext = RuleContext.EMPTY;if ( outerContext.parent==null || outerContext==RuleContext.EMPTY ) {return PredictionContext.EMPTY;}PredictionContext parent = EMPTY;parent = PredictionContext.fromRuleContext(atn, outerContext.parent);ATNState state = atn.states.get(outerContext.invokingState);RuleTransition transition = (RuleTransition)state.transition(0);return SingletonPredictionContext.create(parent, transition.followState.stateNumber);}
public boolean isAbsolute() {return absolute;}
public String toString() {StringBuilder sb = new StringBuilder();sb.append("[SXPI]\n");for (int i = 0; i < _fieldInfos.length; i++) {sb.append("    item[").append(i).append("]=");_fieldInfos[i].appendDebugInfo(sb);sb.append('\n');}sb.append("[/SXPI]\n");return sb.toString();}
public final CharsetDecoder reset() {status = INIT;implReset();return this;}
public TopMarginRecord( RecordInputStream in ) {field_1_margin = in.readDouble();}
public GetDiskResult getDisk(GetDiskRequest request) {request = beforeClientExecution(request);return executeGetDisk(request);}
public DeleteAccountAliasRequest(String accountAlias) {setAccountAlias(accountAlias);}
public String toString() {return "SPL";}
public FtCblsSubRecord() {reserved = new byte[ENCODED_SIZE];}
public void setReadonly(boolean readonly) {if ( this.readonly && !readonly ) throw new IllegalStateException("can't alter readonly IntervalSet");this.readonly = readonly;}
public DeleteQueueResult deleteQueue(String queueUrl) {return deleteQueue(new DeleteQueueRequest().withQueueUrl(queueUrl));}
public CharSequence getFully(CharSequence key) {StringBuilder result = new StringBuilder(tries.size() * 2);for (int i = 0; i < tries.size(); i++) {CharSequence r = tries.get(i).getFully(key);if (r == null || (r.length() == 1 && r.charAt(0) == EOM)) {return result;}result.append(r);}return result;}
public BatchDeleteScheduledActionResult batchDeleteScheduledAction(BatchDeleteScheduledActionRequest request) {request = beforeClientExecution(request);return executeBatchDeleteScheduledAction(request);}
public Merger newMerger(Repository db) {return new OneSide(db, treeIndex);}
public String getName() {return "resolve"; }
public CreateDistributionWithTagsResult createDistributionWithTags(CreateDistributionWithTagsRequest request) {request = beforeClientExecution(request);return executeCreateDistributionWithTags(request);}
public Record getNext() {if(!hasNext()) {throw new RuntimeException("Attempt to read past end of record stream");}_countRead ++;return _list.get(_nextIndex++);}
public float currentScore(int docId, String field, int start, int end, int numPayloadsSeen, float currentScore, float currentPayloadScore) {if (numPayloadsSeen == 0) {return currentPayloadScore;} else {return Math.min(currentPayloadScore, currentScore);}}
public FloatBuffer slice() {byteBuffer.limit(limit * SizeOf.FLOAT);byteBuffer.position(position * SizeOf.FLOAT);ByteBuffer bb = byteBuffer.slice().order(byteBuffer.order());FloatBuffer result = new FloatToByteBufferAdapter(bb);byteBuffer.clear();return result;}
public DescribeVoicesResult describeVoices(DescribeVoicesRequest request) {request = beforeClientExecution(request);return executeDescribeVoices(request);}
public boolean matches(int symbol, int minVocabSymbol, int maxVocabSymbol) {return label == symbol;}
public UpdateHITTypeOfHITResult updateHITTypeOfHIT(UpdateHITTypeOfHITRequest request) {request = beforeClientExecution(request);return executeUpdateHITTypeOfHIT(request);}
public BooleanQuery build(QueryNode queryNode) throws QueryNodeException {AnyQueryNode andNode = (AnyQueryNode) queryNode;BooleanQuery.Builder bQuery = new BooleanQuery.Builder();List<QueryNode> children = andNode.getChildren();if (children != null) {for (QueryNode child : children) {Object obj = child.getTag(QueryTreeBuilder.QUERY_TREE_BUILDER_TAGID);if (obj != null) {Query query = (Query) obj;try {bQuery.add(query, BooleanClause.Occur.SHOULD);} catch (TooManyClauses ex) {throw new QueryNodeException(new MessageImpl(QueryParserMessages.EMPTY_MESSAGE), ex);}}}}bQuery.setMinimumNumberShouldMatch(andNode.getMinimumMatchingElements());return bQuery.build();}
public boolean equals(Object o) {if (this == o) return true;if (o == null || getClass() != o.getClass()) return false;FacetEntry that = (FacetEntry) o;if (count != that.count) return false;if (!value.equals(that.value)) return false;return true;}
public void setProgressMonitor(ProgressMonitor pm) {progressMonitor = pm;}
public MergeShardsResult mergeShards(MergeShardsRequest request) {request = beforeClientExecution(request);return executeMergeShards(request);}
public ByteBuffer read(int length, long position) throws IOException {if(position >= size()) {throw new IndexOutOfBoundsException("Position " + position + " past the end of the file");}ByteBuffer dst;if (writable) {dst = channel.map(FileChannel.MapMode.READ_WRITE, position, length);buffersToClean.add(dst);} else {channel.position(position);dst = ByteBuffer.allocate(length);int worked = IOUtils.readFully(channel, dst);if(worked == -1) {throw new IndexOutOfBoundsException("Position " + position + " past the end of the file");}}dst.position(0);return dst;}
public synchronized void setMultiValued(String dimName, boolean v) {DimConfig ft = fieldTypes.get(dimName);if (ft == null) {ft = new DimConfig();fieldTypes.put(dimName, ft);}ft.multiValued = v;}
public AttributeDefinition(String attributeName, ScalarAttributeType attributeType) {setAttributeName(attributeName);setAttributeType(attributeType.toString());}
public void set(int index, long value) {final int o = index >>> 5;final int b = index & 31;final int shift = b << 1;blocks[o] = (blocks[o] & ~(3L << shift)) | (value << shift);}
public RandomSamplingFacetsCollector(int sampleSize, long seed) {super(false);this.sampleSize = sampleSize;this.random = new XORShift64Random(seed);this.sampledDocs = null;}
public RebootNodeResult rebootNode(RebootNodeRequest request) {request = beforeClientExecution(request);return executeRebootNode(request);}
public static long generationFromSegmentsFileName(String fileName) {if (fileName.equals(OLD_SEGMENTS_GEN)) {throw new IllegalArgumentException("\"" + OLD_SEGMENTS_GEN + "\" is not a valid segment file name since 4.0");} else if (fileName.equals(IndexFileNames.SEGMENTS)) {return 0;} else if (fileName.startsWith(IndexFileNames.SEGMENTS)) {return Long.parseLong(fileName.substring(1+IndexFileNames.SEGMENTS.length()),Character.MAX_RADIX);} else {throw new IllegalArgumentException("fileName \"" + fileName + "\" is not a segments file");}}
public ListObjectPoliciesResult listObjectPolicies(ListObjectPoliciesRequest request) {request = beforeClientExecution(request);return executeListObjectPolicies(request);}
public ResetImageAttributeRequest(String imageId, ResetImageAttributeName attribute) {setImageId(imageId);setAttribute(attribute.toString());}
public BatchRefUpdate setRefLogIdent(PersonIdent pi) {refLogIdent = pi;return this;}
public GetBaiduChannelResult getBaiduChannel(GetBaiduChannelRequest request) {request = beforeClientExecution(request);return executeGetBaiduChannel(request);}
public CreateVpnConnectionResult createVpnConnection(CreateVpnConnectionRequest request) {request = beforeClientExecution(request);return executeCreateVpnConnection(request);}
public boolean hasPrevious() {return link != list.voidLink;}
public FetchLibrariesRequest() {super("CloudPhoto", "2017-07-11", "FetchLibraries", "cloudphoto");setProtocol(ProtocolType.HTTPS);}
public RevFilter clone() {return new PatternSearch(pattern());}
public Token recoverInline(Parser recognizer)throws RecognitionException{InputMismatchException e = new InputMismatchException(recognizer);for (ParserRuleContext context = recognizer.getContext(); context != null; context = context.getParent()) {context.exception = e;}throw new ParseCancellationException(e);}
public static void fill(double[] array, double value) {for (int i = 0; i < array.length; i++) {array[i] = value;}}
public String toString(){final StringBuilder b = new StringBuilder();final int          l = length();b.append(File.separatorChar);for (int i = 0; i < l; i++){b.append(getComponent(i));if (i < l - 1){b.append(File.separatorChar);}}return b.toString();}
public QueryScorer(WeightedSpanTerm[] weightedTerms) {this.fieldWeightedSpanTerms = new HashMap<>(weightedTerms.length);for (int i = 0; i < weightedTerms.length; i++) {WeightedSpanTerm existingTerm = fieldWeightedSpanTerms.get(weightedTerms[i].term);if ((existingTerm == null) ||(existingTerm.weight < weightedTerms[i].weight)) {fieldWeightedSpanTerms.put(weightedTerms[i].term, weightedTerms[i]);maxTermWeight = Math.max(maxTermWeight, weightedTerms[i].getWeight());}}skipInitExtractor = true;}
public BaseRef(AreaEval ae) {_refEval = null;_areaEval = ae;_firstRowIndex = ae.getFirstRow();_firstColumnIndex = ae.getFirstColumn();_height = ae.getLastRow() - ae.getFirstRow() + 1;_width = ae.getLastColumn() - ae.getFirstColumn() + 1;}
public DeleteDBSnapshotRequest(String dBSnapshotIdentifier) {setDBSnapshotIdentifier(dBSnapshotIdentifier);}
public CreateTrafficPolicyInstanceResult createTrafficPolicyInstance(CreateTrafficPolicyInstanceRequest request) {request = beforeClientExecution(request);return executeCreateTrafficPolicyInstance(request);}
public ShortBuffer put(short[] src, int srcOffset, int shortCount) {byteBuffer.limit(limit * SizeOf.SHORT);byteBuffer.position(position * SizeOf.SHORT);if (byteBuffer instanceof ReadWriteDirectByteBuffer) {((ReadWriteDirectByteBuffer) byteBuffer).put(src, srcOffset, shortCount);} else {((ReadWriteHeapByteBuffer) byteBuffer).put(src, srcOffset, shortCount);}this.position += shortCount;return this;}
public StringBuilder replace(int start, int end, String str) {replace0(start, end, str);return this;}
public NumberFormatIndexRecord clone() {return copy();}
public CreateLoadBalancerRequest(String loadBalancerName) {setLoadBalancerName(loadBalancerName);}
public PutVoiceConnectorStreamingConfigurationResult putVoiceConnectorStreamingConfiguration(PutVoiceConnectorStreamingConfigurationRequest request) {request = beforeClientExecution(request);return executePutVoiceConnectorStreamingConfiguration(request);}
public GetLifecyclePolicyPreviewResult getLifecyclePolicyPreview(GetLifecyclePolicyPreviewRequest request) {request = beforeClientExecution(request);return executeGetLifecyclePolicyPreview(request);}
public TokenStream create(TokenStream input) {return new RussianLightStemFilter(input);}
public final void setPrecedenceFilterSuppressed(boolean value) {if (value) {this.reachesIntoOuterContext |= 0x40000000;}else {this.reachesIntoOuterContext &= ~SUPPRESS_PRECEDENCE_FILTER;}}
public V setValue(V object) {throw new UnsupportedOperationException();}
public PutLifecycleEventHookExecutionStatusResult putLifecycleEventHookExecutionStatus(PutLifecycleEventHookExecutionStatusRequest request) {request = beforeClientExecution(request);return executePutLifecycleEventHookExecutionStatus(request);}
public ShortBuffer slice() {return new ReadWriteShortArrayBuffer(remaining(), backingArray, offset + position);}
public ActivateEventSourceResult activateEventSource(ActivateEventSourceRequest request) {request = beforeClientExecution(request);return executeActivateEventSource(request);}
public ObjectId getData() {return data;}
public ListDirectoriesResult listDirectories(ListDirectoriesRequest request) {request = beforeClientExecution(request);return executeListDirectories(request);}
public ReportInstanceStatusResult reportInstanceStatus(ReportInstanceStatusRequest request) {request = beforeClientExecution(request);return executeReportInstanceStatus(request);}
public BlockList() {directory = BlockList.<T> newDirectory(256);directory[0] = BlockList.<T> newBlock();tailBlock = directory[0];}
public static void writeUnicodeStringFlagAndData(LittleEndianOutput out, String value) {boolean is16Bit = hasMultibyte(value);out.writeByte(is16Bit ? 0x01 : 0x00);if (is16Bit) {putUnicodeLE(value, out);} else {putCompressedUnicode(value, out);}}
public final String getFullMessage() {byte[] raw = buffer;int msgB = RawParseUtils.tagMessage(raw, 0);if (msgB < 0) {return ""; }return RawParseUtils.decode(guessEncoding(), raw, msgB, raw.length);}
public EnableAlarmActionsResult enableAlarmActions(EnableAlarmActionsRequest request) {request = beforeClientExecution(request);return executeEnableAlarmActions(request);}
public final Charset charset() {return cs;}
public CreateExportJobResult createExportJob(CreateExportJobRequest request) {request = beforeClientExecution(request);return executeCreateExportJob(request);}
public final int getInt(int index) {checkIndex(index, SizeOf.INT);return Memory.peekInt(backingArray, offset + index, order);}
public Object[] toArray() {return a.clone();}
public String toString() {return "<deleted/>";}
public DBCluster restoreDBClusterToPointInTime(RestoreDBClusterToPointInTimeRequest request) {request = beforeClientExecution(request);return executeRestoreDBClusterToPointInTime(request);}
public StringBuffer append(double d) {RealToString.getInstance().appendDouble(this, d);return this;}
public UpdateAssessmentTargetResult updateAssessmentTarget(UpdateAssessmentTargetRequest request) {request = beforeClientExecution(request);return executeUpdateAssessmentTarget(request);}
public String toString() {StringBuilder r = new StringBuilder();r.append("BlameResult: "); r.append(getResultPath());return r.toString();}
public String toString() {StringBuilder sb = new StringBuilder();sb.append("[CHART]\n");sb.append("    .x     = ").append(getX()).append('\n');sb.append("    .y     = ").append(getY()).append('\n');sb.append("    .width = ").append(getWidth()).append('\n');sb.append("    .height= ").append(getHeight()).append('\n');sb.append("[/CHART]\n");return sb.toString();}
public DescribeDashboardResult describeDashboard(DescribeDashboardRequest request) {request = beforeClientExecution(request);return executeDescribeDashboard(request);}
public static boolean[] copyOf(boolean[] original, int newLength) {if (newLength < 0) {throw new NegativeArraySizeException();}return copyOfRange(original, 0, newLength);}
public boolean equals(Object obj) {if (!(obj instanceof FacetLabel)) {return false;}FacetLabel other = (FacetLabel) obj;if (length != other.length) {return false; }for (int i = length - 1; i >= 0; i--) {if (!components[i].equals(other.components[i])) {return false;}}return true;}
public final double[] array() {return protectedArray();}
public PushConnection openPush() throws TransportException {return new TcpPushConnection();}
public GetDomainSuggestionsResult getDomainSuggestions(GetDomainSuggestionsRequest request) {request = beforeClientExecution(request);return executeGetDomainSuggestions(request);}
public char requireChar(Map<String,String> args, String name) {return require(args, name).charAt(0);}
public OrdRange getOrdRange(String dim) {return prefixToOrdRange.get(dim);}
public String getSheetName(int sheetIndex) {return getBoundSheetRec(sheetIndex).getSheetname();}
public AssociateSigninDelegateGroupsWithAccountResult associateSigninDelegateGroupsWithAccount(AssociateSigninDelegateGroupsWithAccountRequest request) {request = beforeClientExecution(request);return executeAssociateSigninDelegateGroupsWithAccount(request);}
public ParseTreePattern compileParseTreePattern(String pattern, int patternRuleIndex) {if ( getTokenStream()!=null ) {TokenSource tokenSource = getTokenStream().getTokenSource();if ( tokenSource instanceof Lexer ) {Lexer lexer = (Lexer)tokenSource;return compileParseTreePattern(pattern, patternRuleIndex, lexer);}}throw new UnsupportedOperationException("Parser can't discover a lexer to use");}
public List<Head> getNextHeads(char c) {if (matches(c)) {return newHeads;}return FileNameMatcher.EMPTY_HEAD_LIST;}
public QueryNode getChild() {return getChildren().get(0);}
public GetPushTemplateResult getPushTemplate(GetPushTemplateRequest request) {request = beforeClientExecution(request);return executeGetPushTemplate(request);}
public CharSequence toQueryString(EscapeQuerySyntax escaper) {if (isDefaultField(this.field)) {return "\"" + getTermEscapeQuoted(escaper) + "\"";} else {return this.field + ":" + "\"" + getTermEscapeQuoted(escaper) + "\"";}}
public void append(RecordBase r){_recs.add(r);}
public void characters(char[] ch, int start, int length) {contents.append(ch, start, length);}
public RandomAccessFile(String fileName, String mode) throws FileNotFoundException {this(new File(fileName), mode);}
public void removeName(String name) {int index = getNameIndex(name);removeName(index);}
public RevFilter clone() {return new Binary(a.clone(), b.clone());}
public RemoveTagsRequest(String resourceId) {setResourceId(resourceId);}
public RevertCommand include(AnyObjectId commit) {return include(commit.getName(), commit);}
public int[] clear() {return bytesStart = null;}
public void close() throws IOException {if (read(skipBuffer) != -1 || actualSize != expectedSize) {throw new CorruptObjectException(MessageFormat.format(JGitText.get().packfileCorruptionDetected,JGitText.get().wrongDecompressedLength));}int used = bAvail - inf.getRemaining();if (0 < used) {onObjectData(src, buf, p, used);use(used);}inf.reset();}
public void setParams(String params) {super.setParams(params);language = country = variant = "";StringTokenizer st = new StringTokenizer(params, ",");if (st.hasMoreTokens())language = st.nextToken();if (st.hasMoreTokens())country = st.nextToken();if (st.hasMoreTokens())variant = st.nextToken();}
public UploadArchiveResult uploadArchive(UploadArchiveRequest request) {request = beforeClientExecution(request);return executeUploadArchive(request);}
public MemFuncPtg(int subExprLen) {field_1_len_ref_subexpression = subExprLen;}
