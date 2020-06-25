import logging
from itertools import islice

logger = logging.getLogger(__name__)


class Extractor:
    def __init__(self, session):
        self.session = session

    def get_rev_doc_map(self, rev_ids, rvprop={'ids', 'user', 'timestamp',
                                               'userid', 'comment', 'content',
                                               'flags', 'size'}):
        if len(rev_ids) == 0:
            return {}

        logger.debug("Building a map of {0} revisions: {1}"
                     .format(len(rev_ids), rev_ids))
        rev_docs = self.query_revisions_by_revids(rev_ids, rvprop=rvprop)

        return {rd['revid']: rd for rd in rev_docs}

    def get_plaintext(self, revids):
        if len(revids) == 0:
            return {}

        logger.debug("Building a map of {0} revisions: {1}"
                     .format(len(revids), revids))

        revids_iter = iter(revids)
        while True:
            batch_ids = list(islice(revids_iter, 0, 50))
            if len(batch_ids) == 0:
                break
            else:
                doc = self.session.get(action='query', prop='extracts',
                                       revids=batch_ids, explaintext='1',
                                       exsectionformat='plain')

                page_doc = doc['query'].get('pages', {}).values()

        # return {rd['pageid']: rd for rd in page_doc}
        return doc

    def query_revisions_by_revids(self, revids, batch=50, **params):
        revids_iter = iter(revids)
        while True:
            batch_ids = list(islice(revids_iter, 0, batch))
            if len(batch_ids) == 0:
                break
            else:
                doc = self.session.get(action='query', prop='revisions',
                                       revids=batch_ids, rvslots='main',
                                       **params)

                for page_doc in doc['query'].get('pages', {}).values():
                    yield from _normalize_revisions(page_doc)

    def get_user_doc_map(self, user_texts,
                         usprop={'groups', 'registration', 'emailable',
                                 'editcount', 'gender'}):
        if len(user_texts) == 0:
            return {}
        logger.debug("Building a map of {0} user.info.docs"
                     .format(len(user_texts)))
        return {ud['name']: ud
                for ud in self.query_users_by_text(user_texts, usprop=usprop)}

    def query_users_by_text(self, user_texts, batch=50, **params):
        user_texts_iter = iter(user_texts)
        while True:
            batch_texts = list(islice(user_texts_iter, 0, batch))
            if len(batch_texts) == 0:
                break
            else:
                doc = self.session.get(action='query', list='users',
                                       ususers=batch_texts, **params)

                for user_doc in doc['query'].get('users', []):
                    yield user_doc

    def get_user_last_revision(self, user_text, rev_timestamp,
                               ucprop={'ids', 'timestamp', 'comment', 'size'}):
        if user_text is None or rev_timestamp is None:
            return None

        logger.debug("Requesting the last revision by {0} from the API"
                     .format(user_text))
        doc = self.session.get(action="query", list="usercontribs",
                               ucuser=user_text, ucprop=ucprop,
                               uclimit=1, ucdir="older",
                               ucstart=(rev_timestamp - 1))

        rev_docs = doc['query']['usercontribs']

        if len(rev_docs) > 0:
            return rev_docs[0]
        else:
            # It's OK to not find a revision here.
            return None

    def get_all_revision_of_page(self, page_id, rvprop={'ids', 'timestamp', 'size', 'userid', 'content'}):
        if page_id is None:
            return None

        logger.debug("Requesting the all revision by page id {0} from the API"
                     .format(page_id))

        is_continue = True
        rev_docs = list()
        rvcontinue = None

        while is_continue:

            doc = self.session.get(action="query", list="allrevisions",
                                   pageids=page_id, arvlimit=50,
                                   arvprop=rvprop, arvcontinue=rvcontinue, arvslots="*")

            rev_docs.append(list(doc['query']['allrevisions']))

            if "continue" in doc:
                rvcontinue = doc['continue']['arvcontinue']
            else:
                is_continue = False

        if len(rev_docs) > 0:
            return rev_docs
        else:
            # It's OK to not find a revision here.
            return None

    def get_all_revision_of_page_prop(self, page_id, rvprop={'ids', 'timestamp', 'size', 'userid', 'content'},
                                      rv_dir="newer",
                                      rv_limit=50, rv_start=None, rvstartid=None, should_continue=True):
        if page_id is None:
            return None

        logger.debug("Requesting the all revision by page id {0} from the API"
                     .format(page_id))

        is_continue = True
        rev_docs = list()
        rvcontinue = None

        while is_continue:
            doc = self.session.get(action="query", prop="revisions",
                                   pageids=page_id, rvlimit=rv_limit, rvdir=rv_dir,
                                   rvprop=rvprop, rvcontinue=rvcontinue, rvstartid=rvstartid, rvslots="*")

            page_doc = doc['query'].get('pages', {'revisions': []}).values()
            rev_docs.append(list(page_doc)[0]['revisions'])

            if should_continue and "continue" in doc:
                rvcontinue = doc['continue']['rvcontinue']
            else:
                is_continue = False

        if len(rev_docs) > 0:
            return rev_docs
        else:
            # It's OK to not find a revision here.
            return None

    def get_all_contrib_user(self, user_id, ucprop={'ids', 'timestamp', 'size', 'sizediff', 'title'}):
        if user_id is None:
            return None

        logger.debug("Requesting the all revision by page id {0} from the API"
                     .format(user_id))

        is_continue = True
        user_contrib = list()
        uccontinue = None

        index = 0

        while is_continue and index <10:

            doc = self.session.get(action="query", list="usercontribs",
                                   ucuserids=user_id, uclimit=50, ucdir="newer",
                                   ucprop=ucprop, uccontinue=uccontinue)

            user_contrib.append(list(doc['query']['usercontribs']))

            if "continue" in doc:
                uccontinue = doc['continue']['uccontinue']
            else:
                is_continue = False
            print("iteraton: ",index)
            index += 1

        if len(user_contrib) > 0:
            return user_contrib
        else:
            # It's OK to not find a revision here.
            return None

    def get_page_creation_doc(self, page_id,
                              rvprop={'ids', 'user', 'timestamp', 'userid',
                                      'comment', 'flags', 'size'}):
        if page_id is None:
            return None

        logger.debug("Requesting creation revision for ({0}) from the API"
                     .format(page_id))
        doc = self.session.get(action="query", prop="revisions",
                               pageids=page_id, rvdir="newer", rvlimit=1,
                               rvprop=rvprop)

        page_doc = doc['query'].get('pages', {'revisions': []}).values()
        rev_docs = list(page_doc)[0]['revisions']

        if len(rev_docs) == 1:
            return rev_docs[0]
        else:
            # This is bad, but it should be handled by the calling funcion
            return None



def _normalize_revisions(page_doc):
    page_meta = {k: v for k, v in page_doc.items()
                 if k != 'revisions'}
    if 'revisions' in page_doc:
        for revision_doc in page_doc['revisions']:
            revision_doc['page'] = page_meta
            yield revision_doc
