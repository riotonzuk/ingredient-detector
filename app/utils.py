import json
import os

# this is required when using coding center
def get_base_url(port):
    info = json.load(open(os.path.join(os.environ['HOME'], ".smc", "info.json"), 'r'))
    project_id = info['project_id']
    base_url = "/%s/port/%s/" % (project_id, port)
    return base_url

def allowed_file(filename, ALLOWED_EXTENSIONS=set(['png', 'jpg', 'jpeg'])):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# function to add correct and or comma
def and_syntax(alist):
    if len(alist) == 1:
        alist = "".join(alist)
        return alist
    elif len(alist) == 2:
        alist = " and ".join(alist)
        return alist
    elif len(alist) > 2:
        alist[-1] = "and " + alist[-1]
        alist = ", ".join(alist)
        return alist
    else:
        return
