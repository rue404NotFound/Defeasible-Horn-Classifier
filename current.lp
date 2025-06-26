%======================================================================
% Defeasible Horn-rule Classifier with Exceptions
%======================================================================

%---------------------------------------------------------------------- 
% 0. TRAIN / TEST 50-50 AUTO-SPLIT
%----------------------------------------------------------------------

row(ID) :- val(ID,a(10),_).  % For 'data-cancer.lp' a(10) is the target label

% User-specified training/test rows
train(ID) :- train_given(ID).
test(ID)  :- test_given(ID).

% Incase no user-defined split is given, auto-split rows 50/50
no_user_split :-
    #count { X : train_given(X) } == 0,
    #count { Y : test_given(Y)  } == 0.

train(ID) :- no_user_split, row(ID), ID \ 2 = 0.
test(ID)  :- no_user_split, row(ID), ID \ 2 = 1.

%---------------------------------------------------------------------- 
% 1. TUNABLE CONSTANTS  (adjust freely when fine-tuning)
%----------------------------------------------------------------------

#const maxD    = 4.      
#const maxE    = 2.      
#const maxBody = 2.      

default_id(0..maxD-1).         % default rule identifiers
exception_id(1..maxE).         % exception identifiers

value(0..1).

% Identify valid attributes, except the class label at index 10
attr(I) :- val(_,a(I),0), val(_,a(I),1), I != 10.
literal(a(I,V)) :- attr(I), value(V).

%---------------------------------------------------------------------- 
% 3. DEFAULT RULE BODIES
%----------------------------------------------------------------------

% Learn body literals for each default rule (except last two fallback rules)
{ default_body(D,L) : literal(L) } >= 1 :- default_id(D), D < maxD-2.

% Force single-literal fallback rule at position maxD-2
1 { default_body(maxD-2,L) : literal(L) } 1.

% empty catch-all (D = maxD-1)
default_exists(maxD-1).
default_body_len(maxD-1,0).

% Prevent contradictory literals in the same rule (avoiding redundancy)
:- default_body(D,a(I,0)), default_body(D,a(I,1)).

% Compute and limit body length of each default rule
default_body_len(D,N) :- default_id(D), N = #count { L : default_body(D,L) }.
:- default_body_len(D,N), N > maxBody.

default_exists(D) :- default_body(D,_).

%---------------------------------------------------------------------- 
% 4. EXCEPTIONS
%----------------------------------------------------------------------

% Learn body literals for exceptions (if any)
{ exception_body(D,E,L) : literal(L) } >= 1 :-
    default_id(D), exception_id(E).

% Count body length of each exception
exception_body_len(D,E,N) :-
    default_id(D), exception_id(E),
    N = #count { L : exception_body(D,E,L) }.
:- exception_body_len(D,E,N), N > maxBody.

% Prevent contradictory literals in the same exception (avoiding redundancy)
:- exception_body(D,E,a(I,0)), exception_body(D,E,a(I,1)).

% Enforce that exceptions must differ from their default
:- default_body(D,L), exception_id(E), not exception_body(D,E,L).

% Ignore trivial exceptions (must be more specific than their default)
:- exception_exists(D,E),
   default_body_len(D,DL),
   exception_body_len(D,E,EL), EL <= DL.   %  ← FIXED operator

exception_exists(D,E) :- exception_body(D,E,_).

%---------------------------------------------------------------------- 
% 5. COVERAGE
%----------------------------------------------------------------------

% A default covers a row if its body matches that row
default_covers(D,ID) :-
    default_exists(D),
    default_body_len(D,N),
    #count { a(I,V) : default_body(D,a(I,V)), val(ID,a(I),V) } = N,
    row(ID).

% An exception covers a row if its body fully matches the row
exception_covers(D,E,ID) :-
    exception_exists(D,E),
    exception_body_len(D,E,N),
    #count { a(I,V) : exception_body(D,E,a(I,V)), val(ID,a(I),V) } = N,
    row(ID).

%---------------------------------------------------------------------- 
% 6. PRIORITIES
%----------------------------------------------------------------------

% Priority increases with rule index (auto-scaled)
default_prio(D,P) :- default_id(D), P = D*10.
exception_prio(D,E,P) :- default_prio(D,P), exception_id(E).

%---------------------------------------------------------------------- 
% 7. FRESH TRAIN-COVERAGE
%----------------------------------------------------------------------

% A row is covered by a higher-priority rule
covered_by_higher(D,ID) :-
    default_id(D), train(ID), default_covers(D2,ID), D2 < D.

% A rule must cover at least one training example that no higher rule does
fresh_training_coverage(D) :-
    train(ID), default_covers(D,ID), not covered_by_higher(D,ID).

:- default_exists(D), not fresh_training_coverage(D), D < maxD-2.

%---------------------------------------------------------------------- 
% 8. APPLICABLE RULES & WINNERS
%----------------------------------------------------------------------

% Track all applicable rules and exceptions for each row
applicable(ID,default,D,P,L) :-
    default_covers(D,ID), default_prio(D,P), default_body_len(D,L).

applicable(ID,exception,D,P,L) :-
    exception_covers(D,E,ID), exception_prio(D,E,P),
    exception_body_len(D,E,L).

% Resolve conflicts: lowest priority, longest body wins
minP(ID,MP) :- row(ID), MP = #min { P : applicable(ID,_,_,P,_) }.
maxLen(ID,ML) :- minP(ID,MP), ML = #max { L : applicable(ID,_,_,MP,L) }.

winning(ID,default,D) :-
    minP(ID,MP), maxLen(ID,ML), applicable(ID,default,D,MP,ML).

winning(ID,exception,D,E) :-
    minP(ID,MP), maxLen(ID,ML),
    exception_covers(D,E,ID), exception_prio(D,E,MP),
    exception_body_len(D,E,ML).

%---------------------------------------------------------------------- 
% 9. MAJORITY CLASS (for fallback & catch-all strategy, see system outline document)
%----------------------------------------------------------------------

% Count labels in training set
label0(N0) :- N0 = #count { ID : train(ID), val(ID,a(10),0) }.
label1(N1) :- N1 = #count { ID : train(ID), val(ID,a(10),1) }.

% Compute majority label
maj_label(0) :- label0(N0), label1(N1), N0 >= N1.
maj_label(1) :- label1(N1), label0(N0), N1 > N0.

%---------------------------------------------------------------------- 
% 10. PREDICTIONS
%----------------------------------------------------------------------

% Assign predictions depending on the winning rule
prediction(ID,1) :- winning(ID,default,D), D < maxD-2.
prediction(ID,0) :- winning(ID,exception,_,_).
prediction(ID,L) :- winning(ID,default,maxD-2), maj_label(L).
prediction(ID,L) :- winning(ID,default,maxD-1), maj_label(L).

handled(ID)   :- prediction(ID,_).
uncovered(ID) :- row(ID), not handled(ID).

%---------------------------------------------------------------------- 
% 11. OPTIMISATION
%----------------------------------------------------------------------

mis(ID) :- train(ID), prediction(ID,P), val(ID,a(10),T), P != T.
#minimize { 1@4,ID : mis(ID) }.
#minimize { 1@3,D  : default_exists(D), D < maxD-1 }.
#minimize { 1@2,D,E: exception_exists(D,E) }.
#minimize { Len@1 : default_body_len(D,Len) }.
#minimize { Len@1 : exception_body_len(D,E,Len) }.

%---------------------------------------------------------------------- 
% 12. METRICS & OUTPUT
%----------------------------------------------------------------------

% Count training/test rows and misclassified rows
trainRows(M)    :- M = #count { ID : train(ID) }.
testRows(M)     :- M = #count { ID : test(ID) }.
totalRows(M)    :- M = #count { ID : row(ID) }.

misCount(N)     :- N = #count { ID : mis(ID) }.
mistest(ID)     :- test(ID), prediction(ID,P), val(ID,a(10),T), P != T.
misTestCount(N) :- N = #count { ID : mistest(ID) }.

% Compute error as parts per 10,000
error_pp10k(P)      :- misCount(N),    trainRows(M), M>0, P = (N*10000)/M.
error_test_pp10k(E) :- misTestCount(N), testRows(M), M>0, E = (N*10000)/M.

% The fallback literal
fallback_body(L) :- default_body(maxD-2,L).

% Output
#show default_body/2.
#show exception_body/3.
#show fallback_body/1.
#show misCount/1.
#show misTestCount/1.
#show trainRows/1.
#show testRows/1.
#show totalRows/1.
#show error_pp10k/1.
#show error_test_pp10k/1.
