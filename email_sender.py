import yagmail

def send_lead_email(lead_data):
    yag = yagmail.SMTP(
        "balajirekavathi125@gmail.com",
        "nuxp vusb qvae wdly")
    
    body = f"""
    new support Request from {lead_data['name']}
    
    name : {lead_data['name']}
    email : {lead_data['email']}
    phone : {lead_data['phone']}
    country : {lead_data['country']}
    requirement : {lead_data['requirement']} """
    
    yag.send( to = "balajirekavathi125@gmail.com",
             subject = "New Support Request", 
             contents = body )