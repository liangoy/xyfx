from service.nlp_service import nlp_service
from service.db_service import mongodb_service


def update_sms_score(pid):
    data=mongodb_service.nbt_p_search_person_message.find({'accountId':int(pid)},{'_id':0,'content':1})
    sms_list=[i['content']for i in data]
    sms_cutted_list=[nlp_service.cut_word(i,size=100)for i in sms_list]
    score = str(nlp_service.get_score(sms_cutted_list))
    mongodb_service.nbt_p_score.update_one({'accountId':int(pid)},{'$set':{'msgScore':score}},upsert=True)
    return score